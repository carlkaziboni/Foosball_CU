#!/usr/bin/env python3
"""
dual_play.py

Two deterministic rule-based teams (yellow vs blue) playing foosball.
Both sides use the same kinematic-hit controller with mirrored kick
directions. Includes all v2 physics fixes (table collision, ball
friction, rotation damping).

Run from the project root:

  Live viewer (macOS requires mjpython):
    mjpython dual_play.py

  Render to MP4 video:
    python3 dual_play.py --mp4

  Render to MP4 with custom path / goal limit:
    python3 dual_play.py --mp4 my_game.mp4 --goals 10
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import mujoco

# --- Paths -----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(SCRIPT_DIR, "foosball_sim", "v2", "foosball_sim.xml")

# --- Simulation settings ---------------------------------------------------
MAX_SECONDS = 120          # Max wall-clock time before reset (live viewer)
REALTIME    = True         # Sync to wall clock in viewer
KICK_SPEED  = 200.0        # Ball Y-velocity applied on a "hit"
DEFLECT_X   = 15.0         # Random X-deflection on hits

# --- Control parameters ----------------------------------------------------
STRIKE_ZONE_Y  = 18.0     # |dy| < this -> trigger full strike
TRACK_ZONE_Y   = 35.0     # |dy| < this -> track ball laterally & wind up
HIT_RADIUS_XY  = 15.0     # Proximity (X-Y plane) to count as a "hit"
WALL_X         = 33.0     # Side walls
GOAL_Y         = 65.0     # Goal line

# --- Rod definitions -------------------------------------------------------
# Each team has 4 rods. "kick_dir" is +1 (yellow kicks toward +Y) or
# -1 (blue kicks toward -Y).

YELLOW_RODS = [
    {
        "name": "goal",  "team": "yellow",
        "body": "y_goal_rod",
        "linear_ctrl": 0,   "rot_ctrl": 1,
        "ctrl_range_linear": (-10.0, 10.0),
        "ctrl_range_rot":    (-2.5, 2.5),
        "guys": ["y_goal_guy1", "y_goal_guy2", "y_goal_guy3"],
        "rod_y": -52.5,
        "kick_dir": 1.0,
        "joint_prefix": "y_goal",
    },
    {
        "name": "def",  "team": "yellow",
        "body": "y_def_rod",
        "linear_ctrl": 2,   "rot_ctrl": 3,
        "ctrl_range_linear": (-20.0, 20.0),
        "ctrl_range_rot":    (-2.5, 2.5),
        "guys": ["y_def_guy1", "y_def_guy2"],
        "rod_y": -37.5,
        "kick_dir": 1.0,
        "joint_prefix": "y_def",
    },
    {
        "name": "mid",  "team": "yellow",
        "body": "y_mid_rod",
        "linear_ctrl": 4,   "rot_ctrl": 5,
        "ctrl_range_linear": (-7.0, 7.0),
        "ctrl_range_rot":    (-2.5, 2.5),
        "guys": ["y_mid_guy1", "y_mid_guy2", "y_mid_guy3",
                 "y_mid_guy4", "y_mid_guy5"],
        "rod_y": -7.5,
        "kick_dir": 1.0,
        "joint_prefix": "y_mid",
    },
    {
        "name": "attack",  "team": "yellow",
        "body": "y_attack_rod",
        "linear_ctrl": 6,   "rot_ctrl": 7,
        "ctrl_range_linear": (-12.0, 12.0),
        "ctrl_range_rot":    (-2.5, 2.5),
        "guys": ["y_attack_guy1", "y_attack_guy2", "y_attack_guy3"],
        "rod_y": 22.5,
        "kick_dir": 1.0,
        "joint_prefix": "y_attack",
    },
]

BLUE_RODS = [
    {
        "name": "goal",  "team": "blue",
        "body": "b_goal_rod",
        "linear_ctrl": 8,   "rot_ctrl": 9,
        "ctrl_range_linear": (-10.0, 10.0),
        "ctrl_range_rot":    (-2.5, 2.5),
        "guys": ["b_goal_guy1", "b_goal_guy2", "b_goal_guy3"],
        "rod_y": 52.5,
        "kick_dir": -1.0,
        "joint_prefix": "b_goal",
    },
    {
        "name": "def",  "team": "blue",
        "body": "b_def_rod",
        "linear_ctrl": 10,  "rot_ctrl": 11,
        "ctrl_range_linear": (-20.0, 20.0),
        "ctrl_range_rot":    (-2.5, 2.5),
        "guys": ["b_def_guy1", "b_def_guy2"],
        "rod_y": 37.5,
        "kick_dir": -1.0,
        "joint_prefix": "b_def",
    },
    {
        "name": "mid",  "team": "blue",
        "body": "b_mid_rod",
        "linear_ctrl": 12,  "rot_ctrl": 13,
        "ctrl_range_linear": (-7.0, 7.0),
        "ctrl_range_rot":    (-2.5, 2.5),
        "guys": ["b_mid_guy1", "b_mid_guy2", "b_mid_guy3",
                 "b_mid_guy4", "b_mid_guy5"],
        "rod_y": 7.5,
        "kick_dir": -1.0,
        "joint_prefix": "b_mid",
    },
    {
        "name": "attack",  "team": "blue",
        "body": "b_attack_rod",
        "linear_ctrl": 14,  "rot_ctrl": 15,
        "ctrl_range_linear": (-12.0, 12.0),
        "ctrl_range_rot":    (-2.5, 2.5),
        "guys": ["b_attack_guy1", "b_attack_guy2", "b_attack_guy3"],
        "rod_y": -22.5,
        "kick_dir": -1.0,
        "joint_prefix": "b_attack",
    },
]

ALL_RODS = YELLOW_RODS + BLUE_RODS


# ===========================================================================
# Setup helpers
# ===========================================================================

def patch_model(model):
    """
    Fix v2 physics issues in-memory:
      1. Selective collision: disable ground plane, table mesh, rod cylinders,
         rod handles, rod rubber bumpers.  KEEP ball, foosman figures, and
         side-wall rubbers so ball physically bounces off walls & players.
      2. Reduce ball joint friction from 20 → 0.5.
      3. Stabilise rotation actuators: armature=1.0, kp=5000, kd=200.
    """
    # 1. Disable ALL, then re-enable ball + foosmen + side walls
    for i in range(model.ngeom):
        model.geom_contype[i] = 0
        model.geom_conaffinity[i] = 0

    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    if ball_geom >= 0:
        model.geom_contype[ball_geom] = 1
        model.geom_conaffinity[ball_geom] = 1

    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and "table_side_rubber" in name:
            model.geom_contype[i] = 1
            model.geom_conaffinity[i] = 1

    for prefix in ["y_goal", "y_def", "y_mid", "y_attack",
                    "b_goal", "b_def", "b_mid", "b_attack"]:
        for g in range(1, 6):
            for suffix in [f"{prefix}_guy{g}", f"{prefix}_guy{g}_visual"]:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, suffix)
                if gid >= 0:
                    model.geom_contype[gid] = 1
                    model.geom_conaffinity[gid] = 1

    # 2. Ball joint limits & friction
    #    Side-wall rubber geoms sit at Z=7.75, ball at Z=1.705 — can't collide.
    #    Hard joint limits on ball_x keep it inside the field (±32).
    bx_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
    if bx_jid >= 0:
        model.jnt_limited[bx_jid] = 1
        model.jnt_range[bx_jid] = [-32.0, 32.0]
        model.dof_frictionloss[model.jnt_dofadr[bx_jid]] = 0.2
    by_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")
    if by_jid >= 0:
        model.jnt_limited[by_jid] = 1
        model.jnt_range[by_jid] = [-70.0, 78.0]
        model.dof_frictionloss[model.jnt_dofadr[by_jid]] = 0.2

    # 3. Rotation joints: moderate damping + armature + tamed actuator gains
    for prefix in ["y_goal", "y_def", "y_mid", "y_attack",
                    "b_goal", "b_def", "b_mid", "b_attack"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{prefix}_rotation")
        if jid >= 0:
            dof = model.jnt_dofadr[jid]
            model.dof_damping[dof] = 50.0
            model.dof_armature[dof] = 1.0
            model.jnt_range[jid] = [-math.pi, math.pi]
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                f"{prefix}_rotation")
        if aid >= 0:
            model.actuator_gainprm[aid, 0] = 5000.0
            model.actuator_biasprm[aid, 0] = 0.0
            model.actuator_biasprm[aid, 1] = -5000.0
            model.actuator_biasprm[aid, 2] = -200.0

    print("  Model patched: selective collision (ball+foosmen+walls), actuators stabilised.")


def resolve_ids(model):
    """Pre-resolve MuJoCo IDs for the ball and all rods."""
    ids = {}
    ids["ball_body"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    ball_y_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")
    ball_x_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
    ids["ball_qpos_y"] = model.jnt_qposadr[ball_y_jnt]
    ids["ball_qpos_x"] = model.jnt_qposadr[ball_x_jnt]
    ids["ball_dof_y"]  = model.jnt_dofadr[ball_y_jnt]
    ids["ball_dof_x"]  = model.jnt_dofadr[ball_x_jnt]

    for rod in ALL_RODS:
        rod["body_id"] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, rod["body"])
        rod["guy_ids"] = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, g)
            for g in rod["guys"]
        ]
        slide_jnt = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT,
            f"{rod['joint_prefix']}_linear")
        rod["slide_qpos_adr"] = model.jnt_qposadr[slide_jnt]

    return ids


# ===========================================================================
# Ball helpers
# ===========================================================================

def get_ball_xy(data, ids):
    pos = data.body(ids["ball_body"]).xpos
    return pos[0], pos[1]


def ball_speed(data, ids):
    vx = data.qvel[ids["ball_dof_x"]]
    vy = data.qvel[ids["ball_dof_y"]]
    return math.sqrt(vx * vx + vy * vy)


def apply_kick(data, ids, direction_y, deflect_x=0.0):
    data.qvel[ids["ball_dof_y"]] += direction_y * KICK_SPEED
    data.qvel[ids["ball_dof_x"]] += deflect_x


def clamp_ball_to_field(data, ids):
    bx = data.qpos[ids["ball_qpos_x"]]
    if abs(bx) > WALL_X:
        data.qpos[ids["ball_qpos_x"]] = np.clip(bx, -WALL_X, WALL_X)
        data.qvel[ids["ball_dof_x"]] *= -0.5


def nudge_stale_ball(data, ids, stale_counter):
    spd = ball_speed(data, ids)
    if spd < 0.5:
        stale_counter += 1
    else:
        stale_counter = 0
    if stale_counter > 200:
        # Push toward a random direction
        data.qvel[ids["ball_dof_y"]] += np.random.choice([-1, 1]) * np.random.uniform(30, 70)
        data.qvel[ids["ball_dof_x"]] += np.random.uniform(-15, 15)
        stale_counter = 0
    return stale_counter


# ===========================================================================
# Rod control
# ===========================================================================

def nearest_guy_x(data, rod, ball_x):
    """Find the foosman on this rod closest to ball_x. Return (x, dx)."""
    best_x, best_d = None, float("inf")
    for gid in rod["guy_ids"]:
        gx = data.body(gid).xpos[0]
        d = abs(gx - ball_x)
        if d < best_d:
            best_d, best_x = d, gx
    return best_x, best_d


def compute_controls(data, ids, rods):
    """
    Compute actuator commands for one team's rods.
    Returns (partial ctrl dict {index: value}, phases dict).

    Blue rods are mounted with euler="0 -90 180" (flipped 180° vs yellow),
    so their strike / wind-up rotation directions are inverted.
    """
    ctrl_map = {}
    ball_x, ball_y = get_ball_xy(data, ids)
    phases = {}

    for rod in rods:
        rod_y = rod["rod_y"]
        dist_y = abs(ball_y - rod_y)

        li = rod["linear_ctrl"]
        ri = rod["rot_ctrl"]
        lo_lin, hi_lin = rod["ctrl_range_linear"]
        lo_rot, hi_rot = rod["ctrl_range_rot"]

        # Blue rods are flipped -> invert rotation direction
        rot_sign = 1.0 if rod["kick_dir"] > 0 else -1.0

        if dist_y < TRACK_ZONE_Y:
            guy_x, _ = nearest_guy_x(data, rod, ball_x)
            current_slide = data.qpos[rod["slide_qpos_adr"]]
            if guy_x is not None:
                x_err = ball_x - guy_x
                target = current_slide + x_err
                ctrl_map[li] = np.clip(target, lo_lin, hi_lin)

            if dist_y < STRIKE_ZONE_Y:
                ctrl_map[ri] = np.clip(rot_sign * 1.5, lo_rot, hi_rot)
                phases[rod["name"]] = "strike"
            else:
                ctrl_map[ri] = np.clip(rot_sign * -1.0, lo_rot, hi_rot)
                phases[rod["name"]] = "track"
        else:
            ctrl_map[li] = 0.0
            ctrl_map[ri] = 0.0
            phases[rod["name"]] = "idle"

    return ctrl_map, phases


def detect_and_kick(data, ids, rods, phases, verbose=False):
    """
    Kinematic hit: if a rod is striking and its nearest foosman is
    within HIT_RADIUS_XY of the ball, apply a velocity impulse in
    that team's kick direction.
    """
    ball_x, ball_y = get_ball_xy(data, ids)

    for rod in rods:
        if phases.get(rod["name"]) != "strike":
            continue
        _, best_dx = nearest_guy_x(data, rod, ball_x)
        rod_dy = abs(ball_y - rod["rod_y"])
        dist_xy = math.sqrt(best_dx ** 2 + rod_dy ** 2)
        if verbose:
            team = rod["team"]
            print(f"    [{team} {rod['name']}] ball=({ball_x:.1f},{ball_y:.1f}) "
                  f"rod_y={rod['rod_y']} guy_dx={best_dx:.1f} "
                  f"dist={dist_xy:.1f} "
                  f"{'HIT' if dist_xy < HIT_RADIUS_XY else 'miss'}")
        if dist_xy < HIT_RADIUS_XY:
            deflect = np.random.uniform(-DEFLECT_X, DEFLECT_X)
            apply_kick(data, ids, direction_y=rod["kick_dir"],
                       deflect_x=deflect)
            return True
    return False


# ===========================================================================
# Reset & goal detection
# ===========================================================================

def reset_sim(model, data, ids):
    """Reset sim; ball starts at true centre (compensate for body offset y=-4)."""
    mujoco.mj_resetData(model, data)
    data.qpos[ids["ball_qpos_x"]] = np.random.uniform(-3, 3)
    # Ball body is at world y=-4, so qpos_y=+4 puts it at world y=0 (true centre)
    data.qpos[ids["ball_qpos_y"]] = 4.0 + np.random.uniform(-1, 1)
    mujoco.mj_forward(model, data)


def check_goal(data, ids):
    """Return 'yellow' if yellow scores (ball past +Y), 'blue' if blue scores."""
    _, by = get_ball_xy(data, ids)
    if by > GOAL_Y:
        return "yellow"
    elif by < -GOAL_Y:
        return "blue"
    return None


# ===========================================================================
# Main loops
# ===========================================================================

def run_live_viewer(model, data, ids, max_goals):
    import mujoco.viewer

    reset_sim(model, data, ids)
    step = 0
    kick_cooldown = 0
    stale = 0
    score = {"yellow": 0, "blue": 0}

    print("Launching MuJoCo viewer ...")
    print("  Camera: left-drag=rotate, right-drag=pan, scroll=zoom, Esc=quit")
    print(f"  Playing to {max_goals} goals\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        viewer.cam.distance = 160
        viewer.cam.lookat[:] = [0, 0, 5]

        wall_start = time.time()

        while viewer.is_running():
            step_start = time.time()

            # --- Both teams compute controls ---
            y_ctrl, y_phases = compute_controls(data, ids, YELLOW_RODS)
            b_ctrl, b_phases = compute_controls(data, ids, BLUE_RODS)

            for idx, val in {**y_ctrl, **b_ctrl}.items():
                data.ctrl[idx] = val

            mujoco.mj_step(model, data)
            clamp_ball_to_field(data, ids)
            stale = nudge_stale_ball(data, ids, stale)
            step += 1

            # Shared kick cooldown — alternate who checks first for fairness
            if kick_cooldown <= 0:
                if step % 2 == 0:
                    teams = [(YELLOW_RODS, y_phases), (BLUE_RODS, b_phases)]
                else:
                    teams = [(BLUE_RODS, b_phases), (YELLOW_RODS, y_phases)]
                for rods, phases in teams:
                    if detect_and_kick(data, ids, rods, phases):
                        kick_cooldown = 100
                        break
            else:
                kick_cooldown -= 1

            # Status line
            if step % 500 == 0:
                bx, by = get_ball_xy(data, ids)
                spd = ball_speed(data, ids)
                elapsed = time.time() - wall_start
                print(f"  step {step:>6d}  sim_t={data.time:.2f}s  wall={elapsed:.1f}s  "
                      f"ball=({bx:.1f},{by:.1f})  speed={spd:.1f}  "
                      f"score: Y {score['yellow']} - {score['blue']} B")

            # Goal check
            goal = check_goal(data, ids)
            if goal:
                score[goal] += 1
                bx, by = get_ball_xy(data, ids)
                print(f"\n  *** {'YELLOW' if goal == 'yellow' else 'BLUE'} SCORES! ***  "
                      f"ball=({bx:.1f},{by:.1f})  "
                      f"Score: Yellow {score['yellow']} - {score['blue']} Blue")
                if score["yellow"] >= max_goals or score["blue"] >= max_goals:
                    winner = "YELLOW" if score["yellow"] >= max_goals else "BLUE"
                    print(f"\n  === {winner} WINS {score['yellow']}-{score['blue']}! ===\n")
                    time.sleep(3)
                    score = {"yellow": 0, "blue": 0}
                else:
                    time.sleep(1.5)
                reset_sim(model, data, ids)
                step = 0
                kick_cooldown = 0
                stale = 0

            viewer.sync()
            if REALTIME:
                sleep = model.opt.timestep - (time.time() - step_start)
                if sleep > 0:
                    time.sleep(sleep)

            if time.time() - wall_start > MAX_SECONDS:
                print(f"\n  Episode timed out ({MAX_SECONDS}s). Resetting ...")
                reset_sim(model, data, ids)
                wall_start = time.time()
                step = 0

    print("\nViewer closed. Done.")


def run_mp4(model, data, ids, output_path, max_goals):
    FPS = 30
    W, H = 640, 480
    MAX_STEPS = 30000
    SKIP = max(1, int(1.0 / (FPS * model.opt.timestep)))

    reset_sim(model, data, ids)
    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.update_scene(data)

    frames = []
    kick_cooldown = 0
    stale = 0
    score = {"yellow": 0, "blue": 0}
    total_goals = 0

    print(f"Rendering (max {MAX_STEPS} steps, playing to {max_goals} goals) ...")

    for step in range(1, MAX_STEPS + 1):
        # --- Both teams compute controls ---
        y_ctrl, y_phases = compute_controls(data, ids, YELLOW_RODS)
        b_ctrl, b_phases = compute_controls(data, ids, BLUE_RODS)

        for idx, val in {**y_ctrl, **b_ctrl}.items():
            data.ctrl[idx] = val

        mujoco.mj_step(model, data)
        clamp_ball_to_field(data, ids)
        stale = nudge_stale_ball(data, ids, stale)

        # Shared kick cooldown — alternate who checks first
        if kick_cooldown <= 0:
            if step % 2 == 0:
                teams = [(YELLOW_RODS, y_phases), (BLUE_RODS, b_phases)]
            else:
                teams = [(BLUE_RODS, b_phases), (YELLOW_RODS, y_phases)]
            for rods, phases in teams:
                if detect_and_kick(data, ids, rods, phases):
                    kick_cooldown = 100
                    break
        else:
            kick_cooldown -= 1

        # Capture frame
        if step % SKIP == 0:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())

        # Goal check
        goal = check_goal(data, ids)
        if goal:
            score[goal] += 1
            total_goals += 1
            print(f"  {'Yellow' if goal == 'yellow' else 'Blue'} scores at step {step}!  "
                  f"Score: Yellow {score['yellow']} - {score['blue']} Blue")

            # Capture a few post-goal frames
            for _ in range(FPS):
                mujoco.mj_step(model, data)
                renderer.update_scene(data)
                frames.append(renderer.render().copy())

            if total_goals >= max_goals:
                break

            reset_sim(model, data, ids)
            kick_cooldown = 0
            stale = 0

        if step % 2000 == 0:
            bx, by = get_ball_xy(data, ids)
            spd = ball_speed(data, ids)
            print(f"  step {step:>5d}  ball=({bx:.1f},{by:.1f})  speed={spd:.1f}  "
                  f"frames={len(frames)}  "
                  f"score: Y {score['yellow']} - {score['blue']} B")

    renderer.close()
    print(f"\nFinal score: Yellow {score['yellow']} - {score['blue']} Blue")
    print(f"Captured {len(frames)} frames.")

    # -- Encode video -------------------------------------------------------
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg"):
        print("Writing MP4 via ffmpeg ...")
        tmp_dir = tempfile.mkdtemp()
        try:
            from PIL import Image
        except ImportError:
            import subprocess as sp2
            sp2.check_call([sys.executable, "-m", "pip", "install",
                            "--quiet", "Pillow"])
            from PIL import Image
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(tmp_dir, f"f_{i:06d}.png"))
        subprocess.check_call([
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", os.path.join(tmp_dir, "f_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            output_path,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp_dir)
    else:
        output_path = output_path.replace(".mp4", ".gif")
        print(f"ffmpeg not found — saving as GIF: {output_path}")
        import imageio
        imageio.mimsave(output_path, frames, duration=1000 // FPS, loop=0)

    mb = os.path.getsize(output_path) / 1e6
    print(f"\nVideo saved: {output_path} ({mb:.2f} MB)")
    print(f"Open with:  open {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Dual-team deterministic foosball controller")
    parser.add_argument("--mp4", type=str, nargs="?", const="dual_play.mp4",
                        help="Render to MP4 instead of live viewer")
    parser.add_argument("--goals", type=int, default=10,
                        help="Number of goals before video ends (default: 10)")
    args = parser.parse_args()

    print(f"Loading model: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    patch_model(model)
    ids = resolve_ids(model)

    if args.mp4:
        run_mp4(model, data, ids, args.mp4, args.goals)
    else:
        run_live_viewer(model, data, ids, args.goals)


if __name__ == "__main__":
    main()
