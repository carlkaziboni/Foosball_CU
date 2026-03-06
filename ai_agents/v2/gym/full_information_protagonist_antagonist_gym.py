import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
import glfw


def _patch_model(model):
    """
    Fix v2 physics issues in-memory (no XML changes):
      1. Disable table-mesh collision so the ball isn't embedded in the table.
      2. Disable collision on all foosman capsule / rod cylinder geoms
         (capsules extend to z~2.5 and physically block ball movement).
      3. Reduce ball joint friction from 20 → 2 (mass=0.1, so 20 gives 200 m/s² drag).
      4. Add damping to rotation joints to stop runaway spinning.
      5. Clamp rotation joint range to ±π.
    """
    # 1. Table-mesh collision off
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    if table_geom >= 0:
        model.geom_contype[table_geom] = 0
        model.geom_conaffinity[table_geom] = 0

    # 2. Disable collision on all rod / foosman geoms
    for prefix in ["y_goal", "y_def", "y_mid", "y_attack",
                    "b_goal", "b_def", "b_mid", "b_attack"]:
        rod_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                      f"{prefix}_rod")
        if rod_geom >= 0:
            model.geom_contype[rod_geom] = 0
            model.geom_conaffinity[rod_geom] = 0
        for i in range(1, 6):
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                  f"{prefix}_guy{i}")
            if g >= 0:
                model.geom_contype[g] = 0
                model.geom_conaffinity[g] = 0
            gv = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   f"{prefix}_guy{i}_visual")
            if gv >= 0:
                model.geom_contype[gv] = 0
                model.geom_conaffinity[gv] = 0

    # 3. Reduce ball joint friction
    for jname in ["ball_x", "ball_y"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            dof = model.jnt_dofadr[jid]
            model.dof_frictionloss[dof] = 2.0

    # 4 & 5. Rotation joints: add damping, limit range
    for prefix in ["y_goal", "y_def", "y_mid", "y_attack",
                    "b_goal", "b_def", "b_mid", "b_attack"]:
        jnt_name = f"{prefix}_rotation"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jid >= 0:
            dof = model.jnt_dofadr[jid]
            model.dof_damping[dof] = 1000.0
            model.jnt_range[jid] = [-math.pi, math.pi]

from ai_agents.v2.gym.mujoco_table_render_mixin import MujocoTableRenderMixin

DIRECTION_CHANGE = 1
TABLE_MAX_Y_DIM = 65
BALL_STOPPED_COUNT_THRESHOLD = 300  # Was 50. Give it time to wind up!
MAX_STEPS = 2000                    # Was 1000. Let the games play out longer.

RODS = ["_goal_", "_def_", "_mid_", "_attack_"]

# ── Virtual-kick parameters ────────────────────────────────────────────────────
# Foosman collision is disabled (capsules block the ball).  Instead, when a
# foosman is close to the ball *and* the rod is rotated past a threshold,
# we inject a velocity impulse into the ball — exactly like dual_play.py.
KICK_RADIUS   = 5.0    # X-Y proximity to trigger a kick
KICK_SPEED    = 8.0    # peak impulse magnitude
KICK_MIN_ROT  = 0.3    # minimum |rotation| angle (rad) to count as a kick
KICK_COOLDOWN = 30     # steps between consecutive kicks

class FoosballEnv( MujocoTableRenderMixin, gym.Env, ):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, antagonist_model=None, play_until_goal=False, verbose_mode=False):
        super(FoosballEnv, self).__init__()

        # Build path to XML file relative to this file's location
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_file_dir, '..', '..', '..')
        xml_file = os.path.join(project_root, 'foosball_sim', 'v2', 'foosball_sim.xml')
        xml_file = os.path.normpath(xml_file)  # Clean up the path

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        _patch_model(self.model)
        self.data = mujoco.MjData(self.model)

        self.simulation_time = 0

        self.num_rods_per_player = 4
        self.num_players = 2
        self.num_rods = self.num_rods_per_player * self.num_players  # Total rods

        self.protagonist_action_size = self.num_rods_per_player * 2  # 8 actions for protagonist
        self.antagonist_action_size = self.num_rods_per_player * 2   # 8 actions for antagonist

        action_high = np.ones(self.protagonist_action_size)
        self.rotation_action_space = spaces.Box(
            low=-2.5 * action_high, high=2.5 * action_high, dtype=np.float32
        )

        self.goal_linear_action_space = spaces.Box(
            low=-10.0 * action_high, high=10.0 * action_high, dtype=np.float32
        )
        self.def_linear_action_space = spaces.Box(
            low=-20.0 * action_high, high=20.0 * action_high, dtype=np.float32
        )
        self.mid_linear_action_space = spaces.Box(
            low=-7.0 * action_high, high=7.0 * action_high, dtype=np.float32
        )
        self.attack_linear_action_space = spaces.Box(
            low=-12.0 * action_high, high=12.0 * action_high, dtype=np.float32
        )

        # TEMP
        self.action_space = spaces.Box(
            low=-20 * action_high, high=20 * action_high, dtype=np.float32
        )

        obs_dim = 38
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.viewer = None

        self._healthy_reward = 1.0
        self._ctrl_cost_weight = 0.005
        self._terminate_when_unhealthy = True
        self._healthy_z_range = (-80, 80)
        self.max_no_progress_steps = 500

        self.prev_ball_y = None
        self.no_progress_steps = 0
        self.ball_stopped_count = 0
        self._kick_cooldown = 0          # virtual-kick cooldown counter

        self.antagonist_model = antagonist_model
        self.play_until_goal = play_until_goal
        self.verbose_mode = verbose_mode

    def set_antagonist_model(self, antagonist_model):
        self.antagonist_model = antagonist_model

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_x')
        ball_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_y')

        x_qpos_adr = self.model.jnt_qposadr[ball_x_id]
        y_qpos_adr = self.model.jnt_qposadr[ball_y_id]

        # Ball body pos is (0, -4, 1.705), so qpos_y=4 → world Y≈0 (field centre).
        xy_random = np.random.normal(
            loc=[-0.5, 4.0],
            scale=[0.5, 0.5]
        )

        self.data.qpos[x_qpos_adr] = xy_random[0]
        self.data.qpos[y_qpos_adr] = xy_random[1]

        self.simulation_time = 0
        self.prev_ball_y = self.data.qpos[y_qpos_adr]
        self.no_progress_steps = 0
        self.ball_stopped_count = 0
        self._kick_cooldown = 0

        return self._get_obs(), {}

    def step(self, protagonist_action):
        protagonist_action = np.clip(protagonist_action, self.action_space.low, self.action_space.high)

        antagonist_observation = self._get_antagonist_obs()

        if self.antagonist_model is not None:
            antagonist_result = self.antagonist_model.predict(antagonist_observation)
            # SB3 SAC.predict() returns (action, state) tuple — unwrap it
            antagonist_action = antagonist_result[0] if isinstance(antagonist_result, tuple) else antagonist_result
            antagonist_action = np.clip(antagonist_action, -1.0, 1.0)

            antagonist_action = self._adjust_antagonist_action(antagonist_action)
        else:
            antagonist_action = np.zeros(self.antagonist_action_size)

        self.data.ctrl[:self.protagonist_action_size] = protagonist_action
        self.data.ctrl[self.protagonist_action_size:self.protagonist_action_size + self.antagonist_action_size] = antagonist_action

        mujoco.mj_step(self.model, self.data)
        self._apply_virtual_kicks()   # inject ball impulse if foosman near + rotated
        self.simulation_time += self.model.opt.timestep

        obs = self._get_obs()

        reward = self.compute_reward(protagonist_action)
        terminated = self.terminated

        info = {}

        return obs, reward, terminated, False, info

    def _get_ball_obs(self):
        ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_x')
        ball_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_y')
        x_qpos_adr = self.model.jnt_qposadr[ball_x_id]
        y_qpos_adr = self.model.jnt_qposadr[ball_y_id]
        x_qvel_adr = self.model.jnt_dofadr[ball_x_id]
        y_qvel_adr = self.model.jnt_dofadr[ball_y_id]
        # ball_z joint does not exist in the v2 XML — ball sits on table surface.
        # Return 0.0 for z pos/vel to keep observation space at 38-dim.
        ball_pos = [
            self.data.qpos[x_qpos_adr],
            self.data.qpos[y_qpos_adr],
            0.0
        ]
        ball_vel = [
            self.data.qvel[x_qvel_adr],
            self.data.qvel[y_qvel_adr],
            0.0
        ]

        return ball_pos, ball_vel

    def _get_antagonist_obs(self):
        # Full-information env: antagonist sees the same state as protagonist
        return self._get_obs()

    def _get_obs(self):
        ball_pos, ball_vel = self._get_ball_obs()

        rod_slide_positions = []
        rod_slide_velocities = []
        rod_rotate_positions = []
        rod_rotate_velocities = []

        # Collect observations for both players' rods
        for player in ['y', 'b']:
            for rod in RODS:
                # Linear joints
                slide_joint_name = f"{player}{rod}linear"
                slide_joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, slide_joint_name
                )
                slide_qpos_adr = self.model.jnt_qposadr[slide_joint_id]
                slide_qvel_adr = self.model.jnt_dofadr[slide_joint_id]
                rod_slide_positions.append(self.data.qpos[slide_qpos_adr])
                rod_slide_velocities.append(self.data.qvel[slide_qvel_adr])

                # Rotational joints
                rotate_joint_name = f"{player}{rod}rotation"
                rotate_joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, rotate_joint_name
                )
                rotate_qpos_adr = self.model.jnt_qposadr[rotate_joint_id]
                rotate_qvel_adr = self.model.jnt_dofadr[rotate_joint_id]
                rod_rotate_positions.append(self.data.qpos[rotate_qpos_adr])
                rod_rotate_velocities.append(self.data.qvel[rotate_qvel_adr])

        obs = np.concatenate([
            ball_pos,
            ball_vel,
            rod_slide_positions,
            rod_slide_velocities,
            rod_rotate_positions,
            rod_rotate_velocities
        ])

        assert obs.shape == self.observation_space.shape, (
            f"Observation shape {obs.shape} does not match observation space shape {self.observation_space.shape}"
        )

        return obs

    def _adjust_antagonist_action(self, antagonist_action):
        adjusted_action = -antagonist_action.copy()

        return adjusted_action

    # ── Virtual-kick engine ────────────────────────────────────────────────────
    def _apply_virtual_kicks(self):
        """If a foosman is within KICK_RADIUS of the ball *and* its rod is
        rotated past KICK_MIN_ROT, inject a velocity impulse into the ball.
        Yellow kicks toward +Y (opponent goal), blue toward −Y.
        One kick per step max; shared cooldown."""
        if self._kick_cooldown > 0:
            self._kick_cooldown -= 1
            return

        ball_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
        ball_xy  = self.data.body(ball_bid).xpos[:2]

        bx_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_x')
        by_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_y')
        bx_dof = self.model.jnt_dofadr[bx_jid]
        by_dof = self.model.jnt_dofadr[by_jid]

        for player in ['y', 'b']:
            kick_dir = 1.0 if player == 'y' else -1.0
            for rod in RODS:
                rot_jid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{player}{rod}rotation")
                rot_angle = self.data.qpos[self.model.jnt_qposadr[rot_jid]]
                if abs(rot_angle) < KICK_MIN_ROT:
                    continue

                for g in range(1, 6):
                    gid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, f"{player}{rod}guy{g}")
                    if gid < 0:
                        continue
                    guy_xy = self.data.body(gid).xpos[:2]
                    dist   = np.linalg.norm(ball_xy - guy_xy)
                    if dist < KICK_RADIUS:
                        strength = min(abs(rot_angle), 1.5) / 1.5   # 0→1
                        impulse  = KICK_SPEED * strength * (1.0 - dist / KICK_RADIUS)
                        self.data.qvel[by_dof] += kick_dir * impulse
                        self.data.qvel[bx_dof] += (ball_xy[0] - guy_xy[0]) * 1.5
                        self._kick_cooldown = KICK_COOLDOWN
                        return  # one kick per step

    def euclidean_goal_distance(self, x, y):
        return math.sqrt((x - 0) ** 2 + (y - TABLE_MAX_Y_DIM) ** 2)

    # ── Reward function ────────────────────────────────────────────────────────
    def compute_reward(self, protagonist_action):
        ball_pos, ball_vel = self._get_ball_obs()
        ball_y     = ball_pos[1]                             # qpos_y
        ball_speed = math.sqrt(ball_vel[0]**2 + ball_vel[1]**2)

        # 1. GOALS — dominant signal ±3000 ──────────────────────────────────────
        victory = 3000.0 if ball_y >  TABLE_MAX_Y_DIM else 0.0
        loss    = -3000.0 if ball_y < -TABLE_MAX_Y_DIM else 0.0

        # Ball world-frame XY (for proximity & position rewards)
        ball_bid      = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
        ball_world    = self.data.body(ball_bid).xpos
        ball_world_xy = ball_world[:2]

        # 2. CONTACT PROXIMITY — guides early exploration toward the ball ───────
        contact_reward = 0.0
        for rod in RODS:
            for gn in range(1, 6):
                gid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"y{rod}guy{gn}")
                if gid < 0:
                    continue
                dist = np.linalg.norm(ball_world_xy - self.data.body(gid).xpos[:2])
                if dist < 8.0:
                    contact_reward += 1.0 / (dist + 1.0)
        contact_reward = min(contact_reward, 1.5)  # cap to prevent camping

        # 3. BALL POSITION — continuous gradient toward opponent goal ────────────
        #    ball_world[1] ∈ [-65, 65] → reward ∈ [-1.3, +1.3]
        position_reward = ball_world[1] * 0.02

        # 4. FORWARD VELOCITY — reward ball moving toward +Y ───────────────────
        forward_reward = max(ball_vel[1], 0.0) * 0.5

        # 5. PENALTIES ──────────────────────────────────────────────────────────
        time_penalty     = -0.5
        stagnant_penalty = -1.5 if ball_speed < 0.1 else 0.0
        ctrl_cost        = self.control_cost(protagonist_action)

        reward = (victory + loss
                  + contact_reward + position_reward + forward_reward
                  + time_penalty + stagnant_penalty - ctrl_cost)
        return reward

    @property
    def healthy_reward(self):
        return (
                float(self.is_healthy or self._terminate_when_unhealthy)
                * self._healthy_reward
        )

    def control_cost(self, action):
        # 2-norm
        #control_cost = self._ctrl_cost_weight * np.sum(np.square(action)) * -1.0

        # 1-norm
        control_cost = self._ctrl_cost_weight * np.sum(np.abs(action)) * -1.0

        # L0 norm
        #control_cost = self._ctrl_cost_weight * np.count_nonzero(action) * -1.0

        return control_cost

    @property
    def is_healthy(self):
        # ball_z is always 0 (no z joint), so use ball_y to detect out-of-bounds.
        ball_y = self._get_ball_obs()[0][1]
        min_y, max_y = self._healthy_z_range  # reuses existing ±80 range
        is_healthy = min_y < ball_y < max_y

        return is_healthy

    def _is_ball_moving(self):
        ball_pos, ball_vel = self._get_ball_obs()

        return np.linalg.norm(ball_vel) > 0.05  # Was 0.5. Reward even tiny nudges early on.

    def _determine_progression(self):
        ball_y = self._get_ball_obs()[0][1]

        if self.prev_ball_y is not None:
            if ball_y > self.prev_ball_y:
                self.no_progress_steps = 0
            else:
                self.no_progress_steps += 1

        self.prev_ball_y = ball_y

    @property
    def terminated(self):
        self._determine_progression()

        self.ball_stopped_count = 0 if self._is_ball_moving() else self.ball_stopped_count + 1
        ball_stagnant = self.ball_stopped_count >= BALL_STOPPED_COUNT_THRESHOLD

        over_max_steps = self.simulation_time >= MAX_STEPS

        unhealthy = not self.is_healthy
        no_progress = self.no_progress_steps >= self.max_no_progress_steps

        ball_y = self._get_ball_obs()[0][1]
        ball_x = self._get_ball_obs()[0][0]

        victory = ball_y < -1 * TABLE_MAX_Y_DIM or ball_y > TABLE_MAX_Y_DIM  # Ball in any goal

        if victory:
            print("Victory")
            print(f"Ball x: {ball_x}, Ball y: {ball_y}")

        terminated = (
                unhealthy or (no_progress and not self.play_until_goal) or ball_stagnant or over_max_steps #or victory
        ) if self._terminate_when_unhealthy else False

        if self.verbose_mode and terminated:
            print("Terminated")
            print(f"Unhealthy: {unhealthy}, No progress: {no_progress}, Victory: {victory}, Ball stagnant: {ball_stagnant}")
            print("x: ", ball_x, "y: ", ball_y)
        return terminated