#!/usr/bin/env python3
"""
test_v3_physics.py — v3 rotation-limit visualisation and sanity check.

Run from the project root:
    mjpython test_v3_physics.py          # live viewer (recommended on macOS)
    python3  test_v3_physics.py          # live viewer
    python3  test_v3_physics.py --no-render   # headless assertion check only

What it demonstrates / tests:
  1. SWEEP TEST — each rod is commanded to ±180° (well beyond the ±90° limit).
     The script reads back the actual joint angles and asserts they never exceed
     ±π/2 (≈ ±1.5708 rad).  A PASS / FAIL summary is printed.

  2. KICK DEMO — a simple rule-based controller drives both teams.  The viewer
     lets you watch: foosmen kick, block, and pass without ever flipping
     upside-down.

Controls in the MuJoCo viewer:
  Left-drag = rotate camera   Right-drag = pan   Scroll = zoom   Esc = quit
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

XML_PATH = os.path.join(SCRIPT_DIR, "foosball_sim", "v3", "foosball_sim.xml")

# ---------------------------------------------------------------------------
# v3 constants (must match gym)
# ---------------------------------------------------------------------------
ROT_LIMIT        = math.pi / 2   # ±90° hard limit on all rotation joints
KICK_RADIUS      = 6.0
KICK_SPEED       = 80.0
KICK_MIN_ROT     = 0.3             # minimum |rotation| (rad) — below this rod is idle
KICK_MAX_ROT     = math.pi / 3    # maximum |rotation| (rad, ≈60°) — above this foot
                                   # is too high to reach ball (foosman nearly horizontal)
KICK_COOLDOWN    = 10
GOAL_Y           = 40.5
GOAL_HALF_W      = 4.0
WALL_X           = 24.25
WALL_RESTITUTION = 0.95

ROD_PREFIXES = [
    "y_goal", "y_def", "y_mid", "y_attack",
    "b_goal", "b_def", "b_mid", "b_attack",
]

# ---------------------------------------------------------------------------
# Model patch — identical to ai_agents/v3/gym/_patch_model
# ---------------------------------------------------------------------------

def patch_model(model):
    # 1. Disable all collision, then selectively re-enable
    for i in range(model.ngeom):
        model.geom_contype[i] = 0
        model.geom_conaffinity[i] = 0

    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    if ball_geom >= 0:
        model.geom_contype[ball_geom]    = 1
        model.geom_conaffinity[ball_geom] = 1

    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and "table_side_rubber" in name:
            model.geom_contype[i]    = 1
            model.geom_conaffinity[i] = 1

    for prefix in ROD_PREFIXES:
        for g in range(1, 6):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_guy{g}")
            if gid >= 0:
                model.geom_contype[gid]    = 1
                model.geom_conaffinity[gid] = 1

    # 2. Ball joint limits & friction
    for jname, lo, hi in [("ball_x", -24.25, 24.25), ("ball_y", -45.0, 50.0)]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            model.jnt_limited[jid]                   = 1
            model.jnt_range[jid]                     = [lo, hi]
            model.dof_frictionloss[model.jnt_dofadr[jid]] = 0.2

    # 3. Rotation joints: damping + armature + ±ROT_LIMIT + tamed PD gains
    for prefix in ROD_PREFIXES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_rotation")
        if jid >= 0:
            dof = model.jnt_dofadr[jid]
            model.dof_damping[dof]  = 50.0
            model.dof_armature[dof] = 1.0
            model.jnt_limited[jid]  = 1
            model.jnt_range[jid]    = [-ROT_LIMIT, ROT_LIMIT]   # ← v3 key change

        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_rotation")
        if aid >= 0:
            model.actuator_gainprm[aid, 0]  = 5000.0
            model.actuator_biasprm[aid, 0]  = 0.0
            model.actuator_biasprm[aid, 1]  = -5000.0
            model.actuator_biasprm[aid, 2]  = -200.0
            model.actuator_ctrlrange[aid, 0] = -ROT_LIMIT
            model.actuator_ctrlrange[aid, 1] =  ROT_LIMIT


# ---------------------------------------------------------------------------
# ID caching
# ---------------------------------------------------------------------------

def resolve_ids(model):
    ids = {}
    ids["ball_body"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    bx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
    by = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")
    ids["ball_qpos_x"] = model.jnt_qposadr[bx]
    ids["ball_qpos_y"] = model.jnt_qposadr[by]
    ids["ball_dof_x"]  = model.jnt_dofadr[bx]
    ids["ball_dof_y"]  = model.jnt_dofadr[by]

    # Per-rod: actuator indices, joint qpos addrs, guy body ids
    ids["rods"] = []
    # ctrl layout: y_goal(lin=0,rot=1), y_def(2,3), y_mid(4,5), y_attack(6,7),
    #              b_goal(8,9), b_def(10,11), b_mid(12,13), b_attack(14,15)
    rod_defs = [
        ("y_goal",   "yellow",  0,  1, (-8.25,  8.25), -33.25,  1.0,
         ["y_goal_guy1"]),
        ("y_def",    "yellow",  2,  3, (-9.25,  9.25), -23.75,  1.0,
         ["y_def_guy1",   "y_def_guy2",   "y_def_guy3"]),
        ("y_mid",    "yellow",  4,  5, (-8.95,  8.95),  -4.75,  1.0,
         ["y_mid_guy1",   "y_mid_guy2",   "y_mid_guy3", "y_mid_guy4"]),
        ("y_attack", "yellow",  6,  7, (-9.25,  9.25),  14.25,  1.0,
         ["y_attack_guy1","y_attack_guy2","y_attack_guy3"]),
        ("b_goal",   "blue",    8,  9, (-8.25,  8.25),  33.25, -1.0,
         ["b_goal_guy1"]),
        ("b_def",    "blue",   10, 11, (-9.25,  9.25),  23.75, -1.0,
         ["b_def_guy1",   "b_def_guy2",   "b_def_guy3"]),
        ("b_mid",    "blue",   12, 13, (-8.95,  8.95),   4.75, -1.0,
         ["b_mid_guy1",   "b_mid_guy2",   "b_mid_guy3", "b_mid_guy4"]),
        ("b_attack", "blue",   14, 15, (-9.25,  9.25), -14.25, -1.0,
         ["b_attack_guy1","b_attack_guy2","b_attack_guy3"]),
    ]
    for prefix, team, li, ri, lin_range, rod_y, kick_dir, guys in rod_defs:
        rot_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_rotation")
        sli_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_linear")
        ids["rods"].append({
            "prefix":       prefix,
            "team":         team,
            "linear_ctrl":  li,
            "rot_ctrl":     ri,
            "lin_range":    lin_range,
            "rod_y":        rod_y,
            "kick_dir":     kick_dir,
            "rot_qpos":     model.jnt_qposadr[rot_jnt],
            "sli_qpos":     model.jnt_qposadr[sli_jnt],
            "guy_ids": [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, g)
                for g in guys
            ],
        })
    return ids


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def get_ball_world(data, ids):
    pos = data.body(ids["ball_body"]).xpos
    return pos[0], pos[1]


def apply_wall_bounces(data, ids):
    bx_world, by_world = get_ball_world(data, ids)
    vx = data.qvel[ids["ball_dof_x"]]
    vy = data.qvel[ids["ball_dof_y"]]

    if bx_world > WALL_X and vx > 0:
        data.qvel[ids["ball_dof_x"]]  = -abs(vx) * WALL_RESTITUTION
        data.qpos[ids["ball_qpos_x"]] -= (bx_world - WALL_X) + 0.3
    elif bx_world < -WALL_X and vx < 0:
        data.qvel[ids["ball_dof_x"]]  = abs(vx) * WALL_RESTITUTION
        data.qpos[ids["ball_qpos_x"]] += (-WALL_X - bx_world) + 0.3

    if abs(by_world) >= GOAL_Y:
        if abs(bx_world) < GOAL_HALF_W:
            return   # goal opening — let ball through
        if by_world > GOAL_Y and vy > 0:
            data.qvel[ids["ball_dof_y"]]  = -abs(vy) * WALL_RESTITUTION
            data.qpos[ids["ball_qpos_y"]] -= (by_world - GOAL_Y) + 0.5
        elif by_world < -GOAL_Y and vy < 0:
            data.qvel[ids["ball_dof_y"]]  = abs(vy) * WALL_RESTITUTION
            data.qpos[ids["ball_qpos_y"]] += (-GOAL_Y - by_world) + 0.5


def apply_virtual_kicks(data, ids, kick_cooldown):
    if kick_cooldown > 0:
        return kick_cooldown - 1
    bx, by = get_ball_world(data, ids)
    ball_xy = np.array([bx, by])
    for rod in ids["rods"]:
        rot = data.qpos[rod["rot_qpos"]]
        abs_rot = abs(rot)
        # Only kick during the valid swing window — foot must be low enough to reach ball
        if abs_rot < KICK_MIN_ROT or abs_rot > KICK_MAX_ROT:
            continue
        for gid in rod["guy_ids"]:
            gxy = data.body(gid).xpos[:2]
            dist = np.linalg.norm(ball_xy - gxy)
            if dist < KICK_RADIUS:
                strength = min(abs_rot, KICK_MAX_ROT) / KICK_MAX_ROT
                impulse  = KICK_SPEED * strength * (1.0 - dist / KICK_RADIUS)
                data.qvel[ids["ball_dof_y"]] += rod["kick_dir"] * impulse
                data.qvel[ids["ball_dof_x"]] += (ball_xy[0] - gxy[0]) * 1.5
                return KICK_COOLDOWN
    return 0


def reset_ball(data, ids):
    data.qpos[ids["ball_qpos_x"]] = np.random.uniform(-3.0,  3.0)
    data.qpos[ids["ball_qpos_y"]] = 4.0 + np.random.uniform(-1.0, 1.0)
    data.qvel[ids["ball_dof_x"]]  = 0.0
    data.qvel[ids["ball_dof_y"]]  = 0.0


# ---------------------------------------------------------------------------
# Rule-based controller (one team)
# ---------------------------------------------------------------------------
STRIKE_ZONE_Y = 14.0
TRACK_ZONE_Y  = 22.0

def compute_controls(data, ids, team_name):
    """Simple tracking + kick controller for one team."""
    bx, by = get_ball_world(data, ids)
    ctrl = {}
    for rod in ids["rods"]:
        if rod["team"] != team_name:
            continue
        li = rod["linear_ctrl"]
        ri = rod["rot_ctrl"]
        lo, hi = rod["lin_range"]
        rot_sign = 1.0 if rod["kick_dir"] > 0 else -1.0
        dy = abs(by - rod["rod_y"])

        if dy < TRACK_ZONE_Y:
            # Slide closest guy toward ball X
            best_gx = min(
                (data.body(gid).xpos[0] for gid in rod["guy_ids"]),
                key=lambda gx: abs(gx - bx)
            )
            slide = data.qpos[rod["sli_qpos"]]
            ctrl[li] = float(np.clip(slide + (bx - best_gx), lo, hi))
            if dy < STRIKE_ZONE_Y:
                # Kick: rotate to ROT_LIMIT in attack direction
                ctrl[ri] = float(rot_sign * ROT_LIMIT)
            else:
                # Wind up: slight back-rotation
                ctrl[ri] = float(rot_sign * -0.4)
        else:
            ctrl[li] = 0.0
            ctrl[ri] = 0.0

    return ctrl


# ---------------------------------------------------------------------------
# Phase 1 — Rotation sweep test (headless + visual)
# ---------------------------------------------------------------------------

def run_sweep_test(model, data, ids, viewer=None):
    """
    Command every rotation actuator to ±10 (far beyond ±ROT_LIMIT) for
    SETTLE steps, then read back the joint angles.  All must be within
    ROT_LIMIT ± 1e-3 rad.
    """
    SETTLE = 300   # steps to let the joint reach its target and stabilise
    TOL    = 1e-3  # tolerance: joint limit overshoot allowed (rad)

    print("\n" + "=" * 62)
    print("  PHASE 1 — Rotation Limit Sweep Test")
    print(f"  Commanding rods to ±10 rad (limit = ±{math.degrees(ROT_LIMIT):.0f}°)")
    print("=" * 62)

    results = []

    for direction, label in [(+10.0, "+10 (max)"), (-10.0, "-10 (min)")]:
        mujoco.mj_resetData(model, data)
        reset_ball(data, ids)

        # Send every rotation actuator to `direction`
        for rod in ids["rods"]:
            data.ctrl[rod["rot_ctrl"]] = direction
            data.ctrl[rod["linear_ctrl"]] = 0.0

        for step in range(SETTLE):
            mujoco.mj_step(model, data)
            if viewer:
                viewer.sync()

        # Read back actual angles
        print(f"\n  Command = {label}")
        print(f"  {'Rod':<14} {'Angle (rad)':>12}  {'Angle (°)':>10}  {'Status':>6}")
        print("  " + "-" * 50)
        all_ok = True
        for rod in ids["rods"]:
            angle = data.qpos[rod["rot_qpos"]]
            ok = abs(angle) <= ROT_LIMIT + TOL
            status = "OK" if ok else "FAIL"
            if not ok:
                all_ok = False
            results.append(ok)
            print(f"  {rod['prefix']:<14} {angle:>12.4f}  "
                  f"{math.degrees(angle):>9.1f}°  {status:>6}")

        verdict = "PASS" if all_ok else "FAIL"
        print(f"\n  Direction {direction:+.0f}  →  {verdict}")

    total_pass = sum(results)
    total      = len(results)
    print("\n" + "=" * 62)
    print(f"  Sweep test result: {total_pass}/{total} checks passed  "
          f"({'ALL PASS' if total_pass == total else 'FAILURES DETECTED'})")
    print("=" * 62)
    return total_pass == total


# ---------------------------------------------------------------------------
# Phase 2 — Live gameplay demo
# ---------------------------------------------------------------------------

def run_gameplay_demo(model, data, ids, viewer, num_goals=5):
    """Rule-based dual-team gameplay, rendered in the MuJoCo viewer."""
    print(f"\n  PHASE 2 — Live Gameplay Demo  (first to {num_goals} goals)")
    print("  Watch: foosmen kick but never flip upside-down.\n")

    mujoco.mj_resetData(model, data)
    reset_ball(data, ids)
    mujoco.mj_forward(model, data)

    score          = {"yellow": 0, "blue": 0}
    kick_cooldown  = 0
    step           = 0
    max_rot_seen   = 0.0   # track maximum absolute rotation seen in-game
    stale          = 0

    while viewer.is_running():
        step += 1

        # Controllers for both teams
        y_ctrl = compute_controls(data, ids, "yellow")
        b_ctrl = compute_controls(data, ids, "blue")
        for idx, val in {**y_ctrl, **b_ctrl}.items():
            data.ctrl[idx] = val

        mujoco.mj_step(model, data)
        kick_cooldown = apply_virtual_kicks(data, ids, kick_cooldown)
        apply_wall_bounces(data, ids)

        # Track maximum rotation (should never exceed ROT_LIMIT)
        for rod in ids["rods"]:
            angle = abs(data.qpos[rod["rot_qpos"]])
            if angle > max_rot_seen:
                max_rot_seen = angle

        # Nudge stale ball
        vx = data.qvel[ids["ball_dof_x"]]
        vy = data.qvel[ids["ball_dof_y"]]
        speed = math.sqrt(vx * vx + vy * vy)
        stale = stale + 1 if speed < 0.5 else 0
        if stale > 300:
            data.qvel[ids["ball_dof_y"]] += np.random.choice([-1, 1]) * np.random.uniform(20, 50)
            data.qvel[ids["ball_dof_x"]] += np.random.uniform(-10, 10)
            stale = 0

        # Print status every 500 steps
        if step % 500 == 0:
            bx, by = get_ball_world(data, ids)
            print(f"  step {step:>6d}  ball=({bx:+5.1f},{by:+5.1f})  speed={speed:.1f}  "
                  f"max_rot_seen={math.degrees(max_rot_seen):.1f}°  "
                  f"(limit={math.degrees(ROT_LIMIT):.0f}°)  "
                  f"score Y{score['yellow']}-B{score['blue']}")

        # Goal detection
        bx, by = get_ball_world(data, ids)
        if abs(bx) < GOAL_HALF_W:
            if by > GOAL_Y:
                score["yellow"] += 1
                print(f"\n  *** YELLOW SCORES!  Y{score['yellow']}-B{score['blue']} ***\n")
                time.sleep(1.0)
                mujoco.mj_resetData(model, data)
                reset_ball(data, ids)
                kick_cooldown = stale = 0
                if score["yellow"] >= num_goals:
                    break
            elif by < -GOAL_Y:
                score["blue"] += 1
                print(f"\n  *** BLUE SCORES!  Y{score['yellow']}-B{score['blue']} ***\n")
                time.sleep(1.0)
                mujoco.mj_resetData(model, data)
                reset_ball(data, ids)
                kick_cooldown = stale = 0
                if score["blue"] >= num_goals:
                    break

        viewer.sync()

    print(f"\n  Final score: Yellow {score['yellow']} — Blue {score['blue']}")
    print(f"  Max rotation seen during gameplay: {math.degrees(max_rot_seen):.2f}°  "
          f"(limit = {math.degrees(ROT_LIMIT):.0f}°)")
    limit_respected = max_rot_seen <= ROT_LIMIT + 1e-3
    print(f"  Rotation limit respected: {'YES — PASS' if limit_respected else 'NO — FAIL'}")


# ---------------------------------------------------------------------------
# Headless-only mode
# ---------------------------------------------------------------------------

def run_headless(model, data, ids):
    """Run sweep test and a short gameplay sim with no window."""
    passed = run_sweep_test(model, data, ids, viewer=None)

    # Short gameplay check — 2000 steps, assert rotation limit
    print("\n  Running 2000-step headless gameplay check ...")
    mujoco.mj_resetData(model, data)
    reset_ball(data, ids)
    kick_cooldown = 0
    max_rot = 0.0

    for step in range(2000):
        y_ctrl = compute_controls(data, ids, "yellow")
        b_ctrl = compute_controls(data, ids, "blue")
        for idx, val in {**y_ctrl, **b_ctrl}.items():
            data.ctrl[idx] = val
        mujoco.mj_step(model, data)
        kick_cooldown = apply_virtual_kicks(data, ids, kick_cooldown)
        apply_wall_bounces(data, ids)
        for rod in ids["rods"]:
            max_rot = max(max_rot, abs(data.qpos[rod["rot_qpos"]]))

    print(f"  Max rotation in 2000 steps: {math.degrees(max_rot):.3f}°  "
          f"(limit = {math.degrees(ROT_LIMIT):.0f}°)")
    gameplay_ok = max_rot <= ROT_LIMIT + 1e-3
    print(f"  Gameplay rotation check: {'PASS' if gameplay_ok else 'FAIL'}")

    overall = passed and gameplay_ok
    print("\n" + ("=" * 62))
    print(f"  OVERALL RESULT: {'ALL TESTS PASSED' if overall else 'TESTS FAILED'}")
    print("=" * 62)
    return 0 if overall else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="v3 physics test & visualisation")
    parser.add_argument("--no-render", action="store_true",
                        help="Run headless assertions only (no viewer window)")
    parser.add_argument("--goals", type=int, default=5,
                        help="Goals to play to in the gameplay demo (default: 5)")
    args = parser.parse_args()

    print(f"  Loading v3 model: {XML_PATH}")
    if not os.path.isfile(XML_PATH):
        print(f"  ERROR: XML not found at {XML_PATH}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    patch_model(model)
    ids = resolve_ids(model)

    print(f"  Rotation limit enforced: ±{math.degrees(ROT_LIMIT):.0f}° "
          f"(±{ROT_LIMIT:.4f} rad)")
    print(f"  Total joints: {model.njnt}  |  Total actuators: {model.nu}")

    if args.no_render:
        sys.exit(run_headless(model, data, ids))

    # ── Live viewer path ────────────────────────────────────────────────────
    print("\n  Launching MuJoCo viewer ...")
    print("  Left-drag = rotate  |  Right-drag = pan  |  Scroll = zoom  |  Esc = quit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 180.0
        viewer.cam.elevation = -40.0
        viewer.cam.distance  = 100.0
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        # Phase 1: sweep test (animated)
        passed = run_sweep_test(model, data, ids, viewer=viewer)

        if not viewer.is_running():
            print("  Viewer closed during sweep test.")
            return

        print("\n  Pausing 2 s before gameplay demo ...")
        for _ in range(200):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        # Phase 2: gameplay demo
        run_gameplay_demo(model, data, ids, viewer, num_goals=args.goals)

    print("\n  Done.")


if __name__ == "__main__":
    main()
