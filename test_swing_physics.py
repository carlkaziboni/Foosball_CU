#!/usr/bin/env python3
"""
test_swing_physics.py — Foosball v3 swing / kick physics visualiser.

Every rod swings to kick the ball whenever it gets close enough.
Use this to verify:
  • Rods swing correctly and return to upright
  • Ball is struck and moves realistically
  • Foosmen never flip upside-down (hard limit ±90°)

Run:
    mjpython test_swing_physics.py          (macOS — recommended)
    python3  test_swing_physics.py
    python3  test_swing_physics.py --speed 0.5   # slow-motion (0.5x real time)
    python3  test_swing_physics.py --goals 20    # play to 20 goals

Controls in the MuJoCo viewer:
  Left-drag = rotate camera   Right-drag = pan   Scroll = zoom   Esc = quit
"""

import os, sys, math, time, argparse
import numpy as np
import mujoco

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
XML_PATH = os.path.join(SCRIPT_DIR, "foosball_sim", "v3", "foosball_sim.xml")

# ── Physics constants ─────────────────────────────────────────────────────────
ROT_LIMIT        = math.pi / 2   # ±90° hard joint limit (v3 rule)
SWING_ZONE_Y     = 18.0          # ball must be within this Y-distance of rod to trigger swing
SWING_BACK_ANGLE = -0.5          # rod rests slightly wound-back (rad) when ball is NOT close
KICK_RADIUS      = 6.5           # virtual kick impulse radius
KICK_SPEED       = 90.0          # virtual kick impulse magnitude
KICK_MIN_ROT     = 0.25          # minimum |rotation| (rad) before kick impulse fires
KICK_MAX_ROT     = math.pi / 3   # ≈60° — above this the foot is too high (foosman horizontal)
KICK_COOLDOWN    = 8             # steps between successive kick impulses
WALL_X           = 24.25
WALL_Y_BACK      = 45.0          # back wall
GOAL_HALF_W      = 4.0
GOAL_Y           = 40.5
WALL_RESTITUTION = 0.92

ROD_DEFS = [
    # (prefix,    team,     lin_ctrl, rot_ctrl, lin_range,         rod_y,   kick_dir, foosman_bodies)
    ("y_goal",   "yellow",  0,  1,  (-8.25,  8.25), -33.25,  1.0, ["y_goal_guy1"]),
    ("y_def",    "yellow",  2,  3,  (-9.25,  9.25), -23.75,  1.0, ["y_def_guy1",    "y_def_guy2",    "y_def_guy3"]),
    ("y_mid",    "yellow",  4,  5,  (-8.95,  8.95),  -4.75,  1.0, ["y_mid_guy1",    "y_mid_guy2",    "y_mid_guy3",    "y_mid_guy4"]),
    ("y_attack", "yellow",  6,  7,  (-9.25,  9.25),  14.25,  1.0, ["y_attack_guy1", "y_attack_guy2", "y_attack_guy3"]),
    ("b_goal",   "blue",    8,  9,  (-8.25,  8.25),  33.25, -1.0, ["b_goal_guy1"]),
    ("b_def",    "blue",   10, 11,  (-9.25,  9.25),  23.75, -1.0, ["b_def_guy1",    "b_def_guy2",    "b_def_guy3"]),
    ("b_mid",    "blue",   12, 13,  (-8.95,  8.95),   4.75, -1.0, ["b_mid_guy1",    "b_mid_guy2",    "b_mid_guy3",    "b_mid_guy4"]),
    ("b_attack", "blue",   14, 15,  (-9.25,  9.25), -14.25, -1.0, ["b_attack_guy1", "b_attack_guy2", "b_attack_guy3"]),
]


# ── Model patch ───────────────────────────────────────────────────────────────
def patch_model(model):
    # Disable all collisions, then re-enable ball + walls + foosman capsules
    for i in range(model.ngeom):
        model.geom_contype[i] = 0
        model.geom_conaffinity[i] = 0

    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    if ball_id >= 0:
        model.geom_contype[ball_id] = 1
        model.geom_conaffinity[ball_id] = 1

    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and "table_side_rubber" in name:
            model.geom_contype[i] = 1
            model.geom_conaffinity[i] = 1

    for prefix, *_ in ROD_DEFS:
        for g in range(1, 6):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_guy{g}")
            if gid >= 0:
                model.geom_contype[gid] = 1
                model.geom_conaffinity[gid] = 1

    # Ball joint limits + friction
    for jname, lo, hi in [("ball_x", -24.25, 24.25), ("ball_y", -45.0, 50.0)]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            model.jnt_limited[jid] = 1
            model.jnt_range[jid] = [lo, hi]
            model.dof_frictionloss[model.jnt_dofadr[jid]] = 0.15

    # Rotation joints: ±ROT_LIMIT, moderate damping
    for prefix, *_ in ROD_DEFS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_rotation")
        if jid >= 0:
            dof = model.jnt_dofadr[jid]
            model.dof_damping[dof]  = 50.0
            model.dof_armature[dof] = 1.0
            model.jnt_limited[jid]  = 1
            model.jnt_range[jid]    = [-ROT_LIMIT, ROT_LIMIT]
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_rotation")
        if aid >= 0:
            model.actuator_gainprm[aid, 0]   = 5000.0
            model.actuator_biasprm[aid, 0]   = 0.0
            model.actuator_biasprm[aid, 1]   = -5000.0
            model.actuator_biasprm[aid, 2]   = -200.0
            model.actuator_ctrlrange[aid, 0]  = -ROT_LIMIT
            model.actuator_ctrlrange[aid, 1]  =  ROT_LIMIT


# ── ID cache ──────────────────────────────────────────────────────────────────
def resolve_ids(model):
    ids = {}
    ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    bx_jid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
    by_jid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")
    ids["ball_body"]   = ball_body
    ids["ball_qpos_x"] = model.jnt_qposadr[bx_jid]
    ids["ball_qpos_y"] = model.jnt_qposadr[by_jid]
    ids["ball_dof_x"]  = model.jnt_dofadr[bx_jid]
    ids["ball_dof_y"]  = model.jnt_dofadr[by_jid]

    ids["rods"] = []
    for prefix, team, li, ri, lin_range, rod_y, kick_dir, guys in ROD_DEFS:
        rot_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    f"{prefix}_rotation")
        sli_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    f"{prefix}_linear")
        ids["rods"].append({
            "prefix":      prefix,
            "team":        team,
            "lin_ctrl":    li,
            "rot_ctrl":    ri,
            "lin_range":   lin_range,
            "rod_y":       rod_y,
            "kick_dir":    kick_dir,
            "rot_qpos":    model.jnt_qposadr[rot_jid],
            "sli_qpos":    model.jnt_qposadr[sli_jid],
            "guy_ids": [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, g)
                for g in guys
            ],
        })
    return ids


# ── Ball helpers ──────────────────────────────────────────────────────────────
def ball_pos(data, ids):
    p = data.body(ids["ball_body"]).xpos
    return float(p[0]), float(p[1])


def reset_ball(data, ids, vx=0.0, vy=0.0):
    data.qpos[ids["ball_qpos_x"]] = np.random.uniform(-5.0, 5.0)
    data.qpos[ids["ball_qpos_y"]] = np.random.uniform(-5.0, 5.0)
    data.qvel[ids["ball_dof_x"]]  = vx
    data.qvel[ids["ball_dof_y"]]  = vy


def apply_wall_bounces(data, ids):
    bx, by = ball_pos(data, ids)
    vx = data.qvel[ids["ball_dof_x"]]
    vy = data.qvel[ids["ball_dof_y"]]

    if bx >  WALL_X and vx > 0:
        data.qvel[ids["ball_dof_x"]]  = -abs(vx) * WALL_RESTITUTION
        data.qpos[ids["ball_qpos_x"]] -=  (bx - WALL_X) + 0.3
    elif bx < -WALL_X and vx < 0:
        data.qvel[ids["ball_dof_x"]]  =  abs(vx) * WALL_RESTITUTION
        data.qpos[ids["ball_qpos_x"]] += (-WALL_X - bx) + 0.3

    if abs(by) >= GOAL_Y:
        if abs(bx) < GOAL_HALF_W:
            return  # goal mouth — don't bounce
        if by > GOAL_Y and vy > 0:
            data.qvel[ids["ball_dof_y"]]  = -abs(vy) * WALL_RESTITUTION
            data.qpos[ids["ball_qpos_y"]] -=  (by - GOAL_Y) + 0.5
        elif by < -GOAL_Y and vy < 0:
            data.qvel[ids["ball_dof_y"]]  =  abs(vy) * WALL_RESTITUTION
            data.qpos[ids["ball_qpos_y"]] += (-GOAL_Y - by) + 0.5


def apply_virtual_kicks(data, ids, cooldown):
    if cooldown > 0:
        return cooldown - 1
    bx, by = ball_pos(data, ids)
    ball_xy = np.array([bx, by])
    for rod in ids["rods"]:
        rot     = data.qpos[rod["rot_qpos"]]
        abs_rot = abs(rot)
        if abs_rot < KICK_MIN_ROT or abs_rot > KICK_MAX_ROT:
            continue
        for gid in rod["guy_ids"]:
            gxy  = data.body(gid).xpos[:2]
            dist = np.linalg.norm(ball_xy - gxy)
            if dist < KICK_RADIUS:
                strength = abs_rot / KICK_MAX_ROT
                impulse  = KICK_SPEED * strength * (1.0 - dist / KICK_RADIUS)
                data.qvel[ids["ball_dof_y"]] += rod["kick_dir"] * impulse
                data.qvel[ids["ball_dof_x"]] += (ball_xy[0] - gxy[0]) * 1.8
                return KICK_COOLDOWN
    return 0


# ── Swing controller ──────────────────────────────────────────────────────────
def compute_swing_controls(data, ids):
    """
    For every rod:
      • If ball Y is within SWING_ZONE_Y of the rod → slide toward ball X,
        then swing to full kick angle (ROT_LIMIT in kick direction).
      • Otherwise → rest at SWING_BACK_ANGLE (slight wind-back), stay centred.
    """
    bx, by = ball_pos(data, ids)
    ctrl   = {}
    for rod in ids["rods"]:
        li = rod["lin_ctrl"]
        ri = rod["rot_ctrl"]
        lo, hi = rod["lin_range"]
        dy = abs(by - rod["rod_y"])

        if dy < SWING_ZONE_Y:
            # Slide the rod so a foosman lines up with ball X
            current_slide = data.qpos[rod["sli_qpos"]]
            # Pick closest guy to ball
            best_gx = min(
                (data.body(gid).xpos[0] for gid in rod["guy_ids"]),
                key=lambda gx: abs(gx - bx),
            )
            target_slide = float(np.clip(current_slide + (bx - best_gx) * 0.8, lo, hi))
            ctrl[li] = target_slide

            # Swing to full kick angle in team's attack direction
            ctrl[ri] = float(rod["kick_dir"] * ROT_LIMIT)
        else:
            # Ball far away — centre rod, return to slight wind-back
            ctrl[li] = 0.0
            ctrl[ri] = float(rod["kick_dir"] * SWING_BACK_ANGLE)

    return ctrl


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(args):
    if not os.path.isfile(XML_PATH):
        print(f"ERROR: XML not found at {XML_PATH}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    patch_model(model)
    ids = resolve_ids(model)
    mujoco.mj_resetData(model, data)

    # Give ball an initial push so it immediately moves
    reset_ball(data, ids, vx=np.random.uniform(-15, 15), vy=np.random.uniform(-30, 30))
    mujoco.mj_forward(model, data)

    print(f"\n  Foosball v3 — swing physics test")
    print(f"  Rotation limit : ±{math.degrees(ROT_LIMIT):.0f}°")
    print(f"  Swing zone     : ball within {SWING_ZONE_Y} units Y of any rod")
    print(f"  Speed          : {args.speed}x  |  Goals to play : {args.goals}")
    print(f"  Controls: Left-drag=rotate  Right-drag=pan  Scroll=zoom  Esc=quit\n")

    score    = {"yellow": 0, "blue": 0}
    cooldown = 0
    stale    = 0
    step     = 0
    max_rot  = 0.0
    step_dt  = model.opt.timestep   # seconds per sim step

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 90.0    # side view: rotation shows as left/right sweep
        viewer.cam.elevation = -20.0   # gentle downward angle to see the table surface
        viewer.cam.distance  = 130.0
        viewer.cam.lookat[:] = [0.0, 0.0, 2.0]

        while viewer.is_running():
            t0 = time.perf_counter()

            # ── Physics step ────────────────────────────────────────────────
            ctrl = compute_swing_controls(data, ids)
            for idx, val in ctrl.items():
                data.ctrl[idx] = val

            mujoco.mj_step(model, data)
            cooldown = apply_virtual_kicks(data, ids, cooldown)
            apply_wall_bounces(data, ids)

            # Track max rotation seen (should never exceed ROT_LIMIT)
            for rod in ids["rods"]:
                r = abs(data.qpos[rod["rot_qpos"]])
                if r > max_rot:
                    max_rot = r

            # Nudge ball if it stalls
            vx = data.qvel[ids["ball_dof_x"]]
            vy = data.qvel[ids["ball_dof_y"]]
            speed = math.hypot(vx, vy)
            stale = stale + 1 if speed < 0.5 else 0
            if stale > 250:
                data.qvel[ids["ball_dof_y"]] += np.random.choice([-1, 1]) * np.random.uniform(25, 55)
                data.qvel[ids["ball_dof_x"]] += np.random.uniform(-12, 12)
                stale = 0

            # ── Goal detection ───────────────────────────────────────────────
            bx, by = ball_pos(data, ids)
            if abs(bx) < GOAL_HALF_W:
                if by > GOAL_Y:
                    score["yellow"] += 1
                    print(f"  *** YELLOW SCORES!  Y{score['yellow']}-B{score['blue']} ***"
                          f"  (max_rot={math.degrees(max_rot):.1f}°)")
                    time.sleep(0.8)
                    reset_ball(data, ids,
                               vx=np.random.uniform(-10, 10),
                               vy=np.random.uniform(-25, 25))
                    cooldown = stale = 0
                    if score["yellow"] >= args.goals:
                        break
                elif by < -GOAL_Y:
                    score["blue"] += 1
                    print(f"  *** BLUE SCORES!    Y{score['yellow']}-B{score['blue']} ***"
                          f"  (max_rot={math.degrees(max_rot):.1f}°)")
                    time.sleep(0.8)
                    reset_ball(data, ids,
                               vx=np.random.uniform(-10, 10),
                               vy=np.random.uniform(-25, 25))
                    cooldown = stale = 0
                    if score["blue"] >= args.goals:
                        break

            step += 1
            if step % 600 == 0:
                print(f"  step {step:>6d}  ball=({bx:+5.1f},{by:+5.1f})  speed={speed:5.1f}"
                      f"  max_rot={math.degrees(max_rot):.1f}°  "
                      f"score Y{score['yellow']}-B{score['blue']}")

            viewer.sync()

            # Throttle to args.speed × real-time
            elapsed = time.perf_counter() - t0
            target  = step_dt / args.speed
            if elapsed < target:
                time.sleep(target - elapsed)

    print(f"\n  Final score : Yellow {score['yellow']} — Blue {score['blue']}")
    limit_ok = max_rot <= ROT_LIMIT + 1e-3
    print(f"  Max rotation seen : {math.degrees(max_rot):.2f}°  "
          f"(limit = {math.degrees(ROT_LIMIT):.0f}°)  "
          f"→ {'PASS' if limit_ok else 'FAIL — exceeded limit!'}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="v3 swing / kick physics visualiser")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (default 1.0 = real-time; 0.5 = slow-mo)")
    p.add_argument("--goals", type=int, default=10,
                   help="Play until this many goals are scored (default 10)")
    run(p.parse_args())
