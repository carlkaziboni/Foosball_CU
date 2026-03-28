
"""
Hardware test script — pushes slide and kick commands to the Redis task_queue,
which the hardware handler forwards to the Arduinos over serial.

Run the hardware handler first (in a separate terminal):
    python3 servo_control/handler.py    # or however you start it

Then run this script:

    # Dry run — print commands without pushing to Redis:
    python3 -m servo_control.test_hardware --dry-run

    # Live — push to Redis (handler must be running):
    python3 -m servo_control.test_hardware

    # Test only one Arduino's rods:
    python3 -m servo_control.test_hardware --rods 0 1

    # Slower movement (more time for steppers to reach position):
    python3 -m servo_control.test_hardware --delay 1.0
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_control.bridge_config import REDIS_HOST, REDIS_PORT, REDIS_QUEUE, ROD_SPECS

STEP_DELAY = 0.5   # seconds between commands


# ── Redis push helper ──────────────────────────────────────────────────────────

def push(r, msg: dict, dry_run: bool) -> None:
    rod_name = ROD_SPECS[msg["rod"]]["name"]
    tag = "[dry-run] " if dry_run else ""
    print(f"  {tag}rod={msg['rod']} ({rod_name:6s})  {json.dumps(msg)}")
    if not dry_run:
        r.rpush(REDIS_QUEUE, json.dumps(msg))


# ── Test 1: slide sweep ────────────────────────────────────────────────────────

def test_slide_sweep(r, rod_ids: list, dry_run: bool) -> None:
    """Move each rod: min (0.0) → center (0.5) → max (1.0) → center (0.5)."""
    print("\n[1] Slide sweep")
    for rod_id in rod_ids:
        print(f"  Rod {rod_id} ({ROD_SPECS[rod_id]['name']})")
        for ratio, label in [(0.0, "min"), (0.5, "center"), (1.0, "max"), (0.5, "center")]:
            print(f"    {label}")
            push(r, {"rod": rod_id, "type": "slide", "position": ratio}, dry_run)
            time.sleep(STEP_DELAY)



# ── Test 2: kick ───────────────────────────────────────────────────────────────

def test_kick(r, rod_ids: list, dry_run: bool) -> None:
    """Fire a slow kick (fast=0) and a fast kick (fast=1) on each rod."""
    print("\n[2] Kick test")
    for rod_id in rod_ids:
        print(f"  Rod {rod_id} ({ROD_SPECS[rod_id]['name']})")
        for fast, label, angle in [(0, "slow", 45.0), (1, "fast", 90.0)]:
            print(f"    {label} kick  angle={angle}°  fast={fast}")
            push(r, {"rod": rod_id, "type": "kick", "fast": fast, "angle": angle}, dry_run)
            time.sleep(STEP_DELAY)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foosball hardware test via Redis")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without pushing to Redis")
    parser.add_argument("--rods", type=int, nargs="+", default=[0, 1, 2, 3],
                        choices=[0, 1, 2, 3],
                        help="Rod IDs to test (default: 0 1 2 3)")
    parser.add_argument("--delay", type=float, default=STEP_DELAY,
                        help=f"Seconds between commands (default: {STEP_DELAY})")
    parser.add_argument("--kicks-only", action="store_true",
                        help="Skip slide sweep, only run kick test")
    parser.add_argument("--redis-host", default=REDIS_HOST)
    parser.add_argument("--redis-port", type=int, default=REDIS_PORT)
    args = parser.parse_args()

    STEP_DELAY = args.delay

    print("=" * 50)
    print("  Foosball Hardware Test")
    print("=" * 50)
    print(f"  Dry run    : {args.dry_run}")
    print(f"  Rods       : {args.rods}")
    print(f"  Redis      : {args.redis_host}:{args.redis_port}")
    print(f"  Delay      : {args.delay}s between commands")

    r = None
    if not args.dry_run:
        try:
            import redis  # type: ignore[import-untyped]
        except ImportError:
            print("ERROR: pip install redis")
            sys.exit(1)
        r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
        try:
            r.ping()
        except Exception as e:
            print(f"ERROR: Cannot connect to Redis: {e}")
            sys.exit(1)
        print("  Redis      : connected ✓")

    print()
    if not args.kicks_only:
        test_slide_sweep(r, args.rods, args.dry_run)
    test_kick(r, args.rods, args.dry_run)
    print("\nDone.")
