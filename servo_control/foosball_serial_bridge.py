"""
FoosballSerialBridge — translates the RL model's 8-dim continuous action
vector into Redis task_queue messages consumed by the hardware handler.

Message format (JSON, pushed to Redis 'task_queue'):
  Slide: {"rod": rod_id, "type": "slide", "position": ratio}   ratio ∈ [0, 1]
  Kick:  {"rod": rod_id, "type": "kick",  "fast": 0|1, "angle": float_degrees}
         fast=1 when peak rotation indicates a level-2+ kick; angle is the
         peak rotation magnitude converted from radians to degrees.

The hardware handler (servo_control handler script) reads from task_queue and
writes the actual serial bytes to the Arduino.
"""

import json
import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from .bridge_config import (
    KICK_COOLDOWN_STEPS,
    KICK_LEVEL_THRESHOLDS,
    KICK_THRESHOLD_RAD,
    ROD_SPECS,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_QUEUE,
)

log = logging.getLogger(__name__)


class FoosballSerialBridge:
    """
    Converts an 8-dim RL action vector into Redis task_queue messages.

    Args:
        redis_host: Redis server hostname
        redis_port: Redis server port
        dry_run:    if True, log commands but never connect to Redis
    """

    def __init__(
        self,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
        dry_run: bool = False,
    ) -> None:
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._dry_run = dry_run
        self._redis = None

        # Per-rod state for kick detection
        self._kick_cooldown: List[int] = [0, 0, 0, 0]
        self._peak_rot: List[float] = [0.0, 0.0, 0.0, 0.0]

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Connect to Redis."""
        if self._dry_run:
            log.info("[dry-run] Redis connection not opened.")
            return
        try:
            import redis
        except ImportError:
            raise ImportError("redis-py is required: pip install redis")
        self._redis = redis.Redis(host=self._redis_host, port=self._redis_port, db=0)
        self._redis.ping()
        log.info("Connected to Redis at %s:%d", self._redis_host, self._redis_port)

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            self._redis.close()
            self._redis = None
            log.info("Redis connection closed.")

    # ── Main dispatch ──────────────────────────────────────────────────────────

    def send_action(self, action: np.ndarray, prev_rot: np.ndarray) -> None:
        """
        Translate one RL action step into Redis messages.

        Args:
            action:   shape (8,), protagonist action from model.predict().
                      Layout: [slide0, rot0, slide1, rot1, slide2, rot2, slide3, rot3]
            prev_rot: shape (4,), rotation actions from the previous step.
        """
        for i in range(4):
            if self._kick_cooldown[i] > 0:
                self._kick_cooldown[i] -= 1

        for rod_id in range(4):
            slide_action = float(action[rod_id * 2])
            rot_action   = float(action[rod_id * 2 + 1])
            prev         = float(prev_rot[rod_id])

            # Always send slide
            ratio = self._slide_to_ratio(slide_action, rod_id)
            self._publish({"rod": rod_id, "type": "slide", "position": ratio})

            # Conditionally send kick
            should_kick, level = self._detect_kick(rod_id, rot_action, prev)
            if should_kick:
                self._publish({
                    "rod": rod_id,
                    "type": "kick",
                    "fast": 1 if level >= 2 else 0,
                    "angle": round(self._peak_rot[rod_id] * 180.0 / math.pi, 1),
                })
                self._kick_cooldown[rod_id] = KICK_COOLDOWN_STEPS
                self._peak_rot[rod_id] = 0.0

    # ── Conversion helpers ─────────────────────────────────────────────────────

    def _slide_to_ratio(self, slide_cm: float, rod_id: int) -> float:
        """Convert slide position in cm to a ratio [0.0, 1.0]."""
        max_cm = ROD_SPECS[rod_id]["slide_max_cm"]
        slide_cm = max(-max_cm, min(max_cm, slide_cm))
        return (slide_cm + max_cm) / (2.0 * max_cm)

    # ── Kick detection ─────────────────────────────────────────────────────────

    def _detect_kick(
        self, rod_id: int, cur_rot: float, prev_rot: float
    ) -> Tuple[bool, Optional[int]]:
        """
        Fire a kick when the rod is on its downswing past the threshold.

        Conditions:
          1. |cur_rot| >= KICK_THRESHOLD_RAD
          2. |cur_rot| < |prev_rot|  (downswing — magnitude decreasing)
          3. cooldown == 0
        """
        abs_cur  = abs(cur_rot)
        abs_prev = abs(prev_rot)

        if abs_cur > self._peak_rot[rod_id]:
            self._peak_rot[rod_id] = abs_cur

        if (
            abs_cur >= KICK_THRESHOLD_RAD
            and abs_cur < abs_prev
            and self._kick_cooldown[rod_id] == 0
        ):
            level = self._rot_to_level(self._peak_rot[rod_id])
            return True, level

        return False, None

    def _rot_to_level(self, peak_rot: float) -> int:
        for min_rot, lvl in KICK_LEVEL_THRESHOLDS:
            if peak_rot >= min_rot:
                return lvl
        return 1

    # ── Redis I/O ──────────────────────────────────────────────────────────────

    def _publish(self, msg: dict) -> None:
        rod_name = ROD_SPECS[msg["rod"]]["name"]
        if self._dry_run:
            log.debug("[dry-run] %s", json.dumps(msg))
            print(f"  [dry-run] rod={msg['rod']} ({rod_name})  {json.dumps(msg)}")
            return
        self._redis.rpush(REDIS_QUEUE, json.dumps(msg))
