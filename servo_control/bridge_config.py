"""
Configuration constants for the foosball serial bridge.
Edit ARDUINO_PORTS to match the actual /dev/ttyUSBx or /dev/ttyACMx paths
assigned to each Arduino by your OS.
"""

# Rod specs indexed by rod_id 0–3 (matches RL action vector order)
# slide_max_cm: physical linear travel limit (from foosball_sim.xml ctrlrange)
# arduino: which Arduino board handles this rod (0 = PAIR_SELECTOR 0, 1 = PAIR_SELECTOR 1)
ROD_SPECS = [
    {"name": "goal",   "slide_max_cm": 8.25,  "arduino": 0},  # rod 0 — action indices 0,1
    {"name": "def",    "slide_max_cm": 9.25,  "arduino": 0},  # rod 1 — action indices 2,3
    {"name": "mid",    "slide_max_cm": 8.95,  "arduino": 1},  # rod 2 — action indices 4,5
    {"name": "attack", "slide_max_cm": 9.25,  "arduino": 1},  # rod 3 — action indices 6,7
]

# Serial port paths — update these to match your system
# Arduino 0: PAIR_SELECTOR=0, handles rods 0 (goal) and 1 (def)
# Arduino 1: PAIR_SELECTOR=1, handles rods 2 (mid) and 3 (attack)
ARDUINO_PORTS = {
    0: "/dev/ttyUSB0",
    1: "/dev/ttyUSB1",
}

# Must match Arduino Serial.begin() baud rate
BAUD_RATE = 9600

# Redis connection (matches hardware handler defaults)
REDIS_HOST  = "192.168.30.116"
REDIS_PORT  = 6379
REDIS_QUEUE = "task_queue"

# Hardware command rate — how often to push commands to the Arduinos
TARGET_HZ = 30.0

# Kick detection thresholds
# Minimum absolute rotation (rad) to consider a kick intent
KICK_THRESHOLD_RAD = 0.8
# Steps to block another kick on the same rod after one fires
# Matches KICK_COOLDOWN in the sim gym (10 steps)
KICK_COOLDOWN_STEPS = 10
# (min_abs_peak_rot, level) pairs — checked in descending order
KICK_LEVEL_THRESHOLDS = [
    (2.0, 3),
    (1.4, 2),
    (0.8, 1),
]
