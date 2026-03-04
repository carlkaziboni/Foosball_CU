#!/usr/bin/env python3
"""
deterministic_play.py

Deterministic rule-based foosball controller with live MuJoCo 3D viewer.
Each yellow rod tracks the ball laterally and strikes when the ball is within range.

NOTE: The v2 MuJoCo XML has a geometry misalignment: the foosmen capsules
cannot physically reach the ball (z = 6.26 vs z = 1.7).  This script works
around the issue by using a *kinematic hit model*:
  - Rods are moved via the position actuators (smooth animation).
  - When a foosman's body centre is close to the ball in X-Y AND the rod
    is in a "strike" phase, we apply a direct velocity impulse to the ball.
  - Ball deceleration is handled naturally by the ball joints' frictionloss.

Run from the project root:

  Live viewer (macOS requires mjpython):
    mjpython deterministic_play.py

  Render to MP4 video (works with regular python):
    python3 deterministic_play.py --mp4

Controls in the live viewer:
    - Left-click drag   -> rotate camera
    - Right-click drag  -> pan camera
    - Scroll            -> zoom
    - Double-click      -> track a body
    - Esc               -> quit
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
MAX_SECONDS = 60          # Max episode wall-clock time
REALTIME    = True        # Sync to wall clock (False = as fast as possible)
KICK_SPEED  = 200.0       # Ball Y-velocity applied on a "hit"
DEFLECT_X   = 15.0        # Random X-deflection on hits

# --- Rod definitions (yellow protagonist) ----------------------------------
RODS = [
    {
        "name": "goal",
        "body": "y_goal_rod",
        "linear_ctrl": 0,   "rot_ctrl": 1,
        "ctrl_range_linear": (-10.0, 10.0),
        "ctrl_range_rot":    (-2.5, 2.5),
        "guys": ["y_goal_guy1", "y_goal_guy2", "y_goal_guy3"],
        "rod_y": -52.5,
    },
    {
        "name": "def",
        "body": "y_def_rod",
        "linear_ctrl": 2,   "rot_ctrl": 3,
        "ctrl_range_linear": (-20.0, 20.0),
        "ctrl_range_rot":    (-2.5, 2.5),
        "guys": ["y_def_guy1", "y_def_guy2"],
        "rod_y": -37.5,
    },
    {
        "name": "mid",
        "body": "y_mid_rod",
        "linear_ctrl": 4,   "rot_ctrl": 5,
        "ctrl_range_linear": (-7.0, 7.0),
        "ctrl_range_rot":    (-2.5, 2.5),
        "guys": ["y_mid_guy1", "y_mid_guy2", "y_mid_guy3", "y_mid_guy4", "y_mid_guy5"],
        "rod_y": -7.5,
    },
    {
        "name": "attack",
        "body": "y_attack_rod",
        "linear_ctrl": 6,   "rot_ctrl": 7,
        "ctrl_range_linear": (-12.0, 12.0),
        "ctrl_range_rot":    (-2.5, 2.5),
        "guys": ["y_attack_guy1", "y_attack_guy2", "y_attack_guy3"],
        "rod_y": 22.5,
    },
]

# --- Control parameters ----------------------------------------------------
STRIKE_ZONE_Y  = 18.0    # |dy| < this -> trigger full strike
TRACK_ZONE_Y   = 35.0    # |dy| < this -> track ball laterally & wind up
HIT_RADIUS_XY  = 15.0    # Proximity (X-Y plane) to count as a "hit"


# ===========================================================================
# Setup helpers
# ===========================================================================

def patch_model(model):
    """
    Fix v2 physics issues in-memory (no XML changes):
      1. Disable table-mesh collision (ball is embedded, generating huge forces).
      2. Add damping to rotation joints (prevents runaway spinning).
      3. Clamp rotation joint ranges to one revolution.
    """
    # 1. Table-mesh collision off
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    if table_geom >= 0:
        model.geom_contype[table_geom] = 0
        model.geom_conaffinity[table_geom] = 0

    # 1b. Reduce ball joint friction (XML has 20, way too high for m=0.1)
    for jname in ["ball_x", "ball_y"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            dof = model.jnt_dofadr[jid]
            model.dof_frictionloss[dof] = 2.0  # 10x reduction

    # 2 & 3. Rotation joints: add damping, limit range
    for prefix in ["y_goal", "y_def", "y_mid", "y_attack",
                    "b_goal", "b_def", "b_mid", "b_attack"]:
        jnt_name = f"{prefix}_rotation"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jid >= 0:
            dof = model.jnt_dofadr[jid]
            model.dof_damping[dof] = 5000.0
            model.jnt_range[jid] = [-math.pi, math.pi]

    print("  Model patched: table collision off, rotation joints damped & limited.")


def resolve_ids(model):
    """Pre-resolve MuJoCo IDs."""
    ids = {}
    ids["ball_body"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    ball_y_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")
    ball_x_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
    ids["ball_qpos_y"] = model.jnt_qposadr[ball_y_jnt]
    ids["ball_qpos_x"] = model.jnt_qposadr[ball_x_jnt]
    ids["ball_dof_y"] = model.jnt_dofadr[ball_y_jnt]
    ids["ball_dof_x"] = model.jnt_dofadr[ball_x_jnt]

    for rod in RODS:
        rod["body_id"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rod["body"])
        rod["guy_ids"] = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, g)
            for g in rod["guys"]
        ]
        slide_jnt = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"y_{rod['name']}_linear"
        )
        rod["slide_qpos_adr"] = model.jnt_qposadr[slide_jnt]

    return ids


# ===========================================================================
# Ball helpers
# ===========================================================================

def get_ball_xy(data, ids):
    """Return ball world (x, y) — use body xpos, not raw qpos."""
    pos = data.body(ids["ball_body"]).xpos
    return pos[0], pos[1]


def ball_speed(data, ids):
    vx = data.qvel[ids["ball_dof_x"]]
    vy = data.qvel[ids["ball_dof_y"]]
    return math.sqrt(vx * vx + vy * vy)


def apply_kick(data, ids, direction_y=1.0, deflect_x=0.0):
    data.qvel[ids["ball_dof_y"]] += direction_y * KICK_SPEED
    data.qvel[ids["ball_dof_x"]] += deflect_x


# ===========================================================================
# Rod control
# ===========================================================================

def nearest_guy_x(data, rod, ball_x):
    best_x, best_d = None, float("inf")
    for gid in rod["guy_ids"]:
        gx = data.body(gid).xpos[0]
        d = abs(gx - ball_x)
        if d < best_d:
            best_d, best_x = d, gx
    return best_x, best_d


def compute_controls(data, ids):
    """
    For each rod:
      IDLE     -> ball far       -> centre slide, neutral rotation
      TRACK    -> ball coming    -> slide to align, wind-up rotation
      STRIKE   -> ball in range  -> snap rotation forward
    Returns (ctrl array, dict rod_name -> phase)
    """
    ctrl = np.zeros(data.ctrl.shape)
    ball_x, ball_y = get_ball_xy(data, ids)
    phases = {}

    for rod in RODS:
        rod_y = rod["rod_y"]
        dist_y = abs(ball_y - rod_y)

        li = rod["linear_ctrl"]
        ri = rod["rot_ctrl"]
        lo_lin, hi_lin = rod["ctrl_range_linear"]
        lo_rot, hi_rot = rod["ctrl_range_rot"]

        if dist_y < TRACK_ZONE_Y:
            # Align nearest foosman with ball in X
            guy_x, _ = nearest_guy_x(data, rod, ball_x)
            current_slide = data.qpos[rod["slide_qpos_adr"]]
            if guy_x is not None:
                x_err = ball_x - guy_x
                target = current_slide + x_err
                ctrl[li] = np.clip(target, lo_lin, hi_lin)

            if dist_y < STRIKE_ZONE_Y:
                ctrl[ri] = np.clip(1.5, lo_rot, hi_rot)
                phases[rod["name"]] = "strike"
            else:
                ctrl[ri] = np.clip(-1.0, lo_rot, hi_rot)
                phases[rod["name"]] = "track"
        else:
            ctrl[li] = 0.0
            ctrl[ri] = 0.0
            phases[rod["name"]] = "idle"

    return ctrl, phases


def detect_and_kick(data, ids, phases, verbose=False):
    """
    Kinematic hit: if a rod is striking and its nearest foosman is
    within HIT_RADIUS_XY of the ball, apply a velocity impulse.
    """
    ball_x, ball_y = get_ball_xy(data, ids)

    for rod in RODS:
        if phases.get(rod["name"]) != "strike":
            continue
        _, best_dx = nearest_guy_x(data, rod, ball_x)
        rod_dy = abs(ball_y - rod["rod_y"])
        dist_xy = math.sqrt(best_dx ** 2 + rod_dy ** 2)
        if verbose:
            print(f"    [{rod['name']}] ball=({ball_x:.1f},{ball_y:.1f}) "
                  f"rod_y={rod['rod_y']} guy_dx={best_dx:.1f} "
                  f"rod_dy={rod_dy:.1f} dist={dist_xy:.1f} "
                  f"{'HIT' if dist_xy < HIT_RADIUS_XY else 'miss'}")
        if dist_xy < HIT_RADIUS_XY:
            deflect = np.random.uniform(-DEFLECT_X, DEFLECT_X)
            apply_kick(data, ids, direction_y=1.0, deflect_x=deflect)
            if verbose:
                spd = ball_speed(data, ids)
                print(f"      KICK! deflect_x={deflect:.1f} new_speed={spd:.1f}")
            return True
    return False


# ===========================================================================
# Reset & goal
# ===========================================================================

def reset_sim(model, data, ids):
    """Reset sim. Ball starts near midfield where mid rod can immediately hit it."""
    mujoco.mj_resetData(model, data)
    # Place ball between mid rod (y=-7.5) and attack rod (y=22.5)
    data.qpos[ids["ball_qpos_x"]] = np.random.uniform(-3, 3)
    data.qpos[ids["ball_qpos_y"]] = np.random.uniform(-2, 2)  # world y near -4 to 0
    mujoco.mj_forward(model, data)


def clamp_ball_to_field(data, ids):
    """Bounce the ball off side walls (table mesh collision is off)."""
    WALL_X = 33.0   # side walls at approximately +/-33
    bx = data.qpos[ids["ball_qpos_x"]]
    if abs(bx) > WALL_X:
        data.qpos[ids["ball_qpos_x"]] = np.clip(bx, -WALL_X, WALL_X)
        data.qvel[ids["ball_dof_x"]] *= -0.5  # bounce with energy loss


def nudge_stale_ball(data, ids, stale_counter):
    """If ball hasn't moved for too long, give it a push."""
    spd = ball_speed(data, ids)
    if spd < 0.5:
        stale_counter += 1
    else:
        stale_counter = 0
    if stale_counter > 200:  # ~0.4s of no movement
        data.qvel[ids["ball_dof_y"]] += np.random.uniform(20, 60)
        data.qvel[ids["ball_dof_x"]] += np.random.uniform(-10, 10)
        stale_counter = 0
    return stale_counter


def check_goal(data, ids):
    _, by = get_ball_xy(data, ids)
    if by > 65:
        return "scored"
    elif by < -65:
        return "conceded"
    return None


# ===========================================================================
# Main loops
# ===========================================================================

def run_live_viewer(model, data, ids):
    import mujoco.viewer

    reset_sim(model, data, ids)
    step = 0
    kick_cooldown = 0
    stale = 0

    print("Launching MuJoCo viewer ...")
    print("  Camera: left-drag=rotate, right-drag=pan, scroll=zoom")
    print(f"  Episode length: {MAX_SECONDS}s  |  Esc to quit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        viewer.cam.distance = 160
        viewer.cam.lookat[:] = [0, 0, 5]

        wall_start = time.time()

        while viewer.is_running():
            step_start = time.time()

            ctrl, phases = compute_controls(data, ids)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            clamp_ball_to_field(data, ids)
            stale = nudge_stale_ball(data, ids, stale)
            step += 1

            if kick_cooldown <= 0:
                if detect_and_kick(data, ids, phases):
                    kick_cooldown = 100
            else:
                kick_cooldown -= 1

            if step % 500 == 0:
                bx, by = get_ball_xy(data, ids)
                spd = ball_speed(data, ids)
                elapsed = time.time() - wall_start
                print(f"  step {step:>6d}  sim_t={data.time:.2f}s  wall={elapsed:.1f}s  "
                      f"ball=({bx:.1f}, {by:.1f})  speed={spd:.1f}")

            goal = check_goal(data, ids)
            if goal:
                bx, by = get_ball_xy(data, ids)
                tag = "GOAL SCORED!" if goal == "scored" else "GOAL CONCEDED!"
                print(f"\n  *** {tag} ***  ball=({bx:.1f},{by:.1f})")
                time.sleep(2)
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


def run_mp4(model, data, ids, output_path):
    FPS = 30
    W, H = 640, 480
    MAX_STEPS = 15000
    SKIP = max(1, int(1.0 / (FPS * model.opt.timestep)))

    reset_sim(model, data, ids)
    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.update_scene(data)

    frames = []
    kick_cooldown = 0
    goals = 0
    stale = 0

    print(f"Rendering (max {MAX_STEPS} steps, capture every {SKIP}) ...")

    for step in range(1, MAX_STEPS + 1):
        ctrl, phases = compute_controls(data, ids)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        clamp_ball_to_field(data, ids)
        stale = nudge_stale_ball(data, ids, stale)

        if kick_cooldown <= 0:
            if detect_and_kick(data, ids, phases, verbose=False):
                kick_cooldown = 100
        else:
            kick_cooldown -= 1

        if step % SKIP == 0:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())

        goal = check_goal(data, ids)
        if goal:
            tag = "scored" if goal == "scored" else "conceded"
            print(f"  Goal {tag} at step {step}!")
            # Capture a few more frames of the goal moment
            for _ in range(FPS):
                mujoco.mj_step(model, data)
                renderer.update_scene(data)
                frames.append(renderer.render().copy())
            # Reset and continue
            reset_sim(model, data, ids)
            kick_cooldown = 0
            goals += 1
            if goals >= 5:
                break  # Enough goals for the video

        if step % 1000 == 0:
            bx, by = get_ball_xy(data, ids)
            spd = ball_speed(data, ids)
            print(f"  step {step:>5d}  ball=({bx:.1f},{by:.1f})  speed={spd:.1f}  frames={len(frames)}")

    renderer.close()
    print(f"Captured {len(frames)} frames.")

    # -- Encode video -------------------------------------------------------
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg"):
        # Use ffmpeg directly (most reliable)
        print("Writing MP4 via ffmpeg ...")
        tmp_dir = tempfile.mkdtemp()
        try:
            from PIL import Image
        except ImportError:
            import subprocess as sp2
            sp2.check_call([sys.executable, "-m", "pip", "install", "--quiet", "Pillow"])
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
        # Fallback: save as GIF
        output_path = output_path.replace(".mp4", ".gif")
        print(f"ffmpeg not found — saving as GIF: {output_path}")
        import imageio
        imageio.mimsave(output_path, frames, duration=1000 // FPS, loop=0)
    mb = os.path.getsize(output_path) / 1e6
    print(f"\nVideo saved: {output_path} ({mb:.2f} MB)")
    print(f"Open with:  open {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Deterministic foosball controller")
    parser.add_argument("--mp4", type=str, nargs="?", const="deterministic_play.mp4",
                        help="Render to MP4 instead of live viewer")
    args = parser.parse_args()

    print(f"Loading model: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    patch_model(model)
    ids = resolve_ids(model)

    if args.mp4:
        run_mp4(model, data, ids, args.mp4)
    else:
        run_live_viewer(model, data, ids)


if __name__ == "__main__":
    main()
