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
      1. Selective collision: disable ground plane, table mesh, rod cylinders,
         rod handles, and rod rubber bumpers.  KEEP ball, foosman figures, and
         side-wall rubbers so the ball physically bounces off walls and gets
         struck by foosmen.
      2. Reduce ball joint friction from 20 → 0.5.
      3. Stabilise rotation actuators: armature=1.0, kp=5000, kd=200.
    """
    # 1. Disable ALL geom collision first, then re-enable the ones we need
    for i in range(model.ngeom):
        model.geom_contype[i] = 0
        model.geom_conaffinity[i] = 0

    # Re-enable ball
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    if ball_geom >= 0:
        model.geom_contype[ball_geom] = 1
        model.geom_conaffinity[ball_geom] = 1

    # Re-enable side-wall rubber bumpers (keep ball in bounds)
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and "table_side_rubber" in name:
            model.geom_contype[i] = 1
            model.geom_conaffinity[i] = 1

    # Re-enable foosman figure geoms (capsules + meshes) so ball bounces off them
    for prefix in ["y_goal", "y_def", "y_mid", "y_attack",
                    "b_goal", "b_def", "b_mid", "b_attack"]:
        for g in range(1, 6):
            for suffix in [f"{prefix}_guy{g}", f"{prefix}_guy{g}_visual"]:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, suffix)
                if gid >= 0:
                    model.geom_contype[gid] = 1
                    model.geom_conaffinity[gid] = 1

    # 2. Ball joint limits & friction
    #    Side-wall rubber geoms sit at Z=7.75, ball at Z=1.705 — collision can't
    #    contain it.  Hard joint limits on ball_x keep it inside the field (±32).
    bx_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
    if bx_jid >= 0:
        model.jnt_limited[bx_jid] = 1
        model.jnt_range[bx_jid] = [-32.0, 32.0]   # walls at ±33.75
        model.dof_frictionloss[model.jnt_dofadr[bx_jid]] = 0.2
    by_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")
    if by_jid >= 0:
        model.jnt_limited[by_jid] = 1
        model.jnt_range[by_jid] = [-70.0, 78.0]   # generous: goals at world ±65
        model.dof_frictionloss[model.jnt_dofadr[by_jid]] = 0.2

    # 3. Rotation joints: moderate damping, armature for stability, limit range
    #    Original gains (40k–150k) + tiny inertia (0.044) + large ctrl range
    #    → catastrophic instability at DOF 5.  Replace with critically-damped
    #    PD: ωn ≈ 70 rad/s → quarter-turn in 0.023 s.  Stability ratio
    #    kp·dt²/(I+armature) = 5000·4e-6/1.044 = 0.019 (rock-solid).
    for prefix in ["y_goal", "y_def", "y_mid", "y_attack",
                    "b_goal", "b_def", "b_mid", "b_attack"]:
        jnt_name = f"{prefix}_rotation"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jid >= 0:
            dof = model.jnt_dofadr[jid]
            model.dof_damping[dof] = 50.0
            model.dof_armature[dof] = 1.0
            model.jnt_range[jid] = [-math.pi, math.pi]

        # Tame the position-servo actuator: kp=5000, kd=200
        act_name = f"{prefix}_rotation"
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if aid >= 0:
            model.actuator_gainprm[aid, 0] = 5000.0        # kp
            model.actuator_biasprm[aid, 0] = 0.0
            model.actuator_biasprm[aid, 1] = -5000.0       # -kp
            model.actuator_biasprm[aid, 2] = -200.0         # -kd

from ai_agents.v2.gym.mujoco_table_render_mixin import MujocoTableRenderMixin

DIRECTION_CHANGE = 1
TABLE_MAX_Y_DIM = 65
BALL_STOPPED_COUNT_THRESHOLD = 300
MAX_EPISODE_STEPS = 1500              # actual env.step() calls, NOT simulation time

RODS = ["_goal_", "_def_", "_mid_", "_attack_"]

# ── Virtual-kick parameters ────────────────────────────────────────────────────
# Foosman collision is disabled (capsules block the ball).  Instead, when a
# foosman is close to the ball *and* the rod is rotated past a threshold,
# we inject a velocity impulse into the ball — exactly like dual_play.py.
KICK_RADIUS   = 10.0   # X-Y proximity to trigger a kick (foosman Y spacing ~15-30)
KICK_SPEED    = 120.0  # peak impulse magnitude — needs to be large for 130-unit field
KICK_MIN_ROT  = 0.3    # minimum |rotation| angle (rad) to count as a kick
KICK_COOLDOWN = 10     # steps between consecutive kicks

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

        # ── Cache ALL MuJoCo IDs once (avoids ~100+ mj_name2id per step) ──────
        self._cache_ids()

    def set_antagonist_model(self, antagonist_model):
        self.antagonist_model = antagonist_model

    def _cache_ids(self):
        """Pre-resolve every MuJoCo name→id mapping used in step/reward/obs."""
        _jid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        _bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)

        # Ball joints
        bx = _jid('ball_x');  by = _jid('ball_y')
        self._bx_qpos = self.model.jnt_qposadr[bx]
        self._by_qpos = self.model.jnt_qposadr[by]
        self._bx_dof  = self.model.jnt_dofadr[bx]
        self._by_dof  = self.model.jnt_dofadr[by]
        self._ball_bid = _bid('ball')

        # Rod joints: slide + rotation for both players (used by _get_obs)
        self._rod_slide_qpos = []
        self._rod_slide_dof  = []
        self._rod_rot_qpos   = []
        self._rod_rot_dof    = []
        for player in ['y', 'b']:
            for rod in RODS:
                sj = _jid(f"{player}{rod}linear")
                rj = _jid(f"{player}{rod}rotation")
                self._rod_slide_qpos.append(self.model.jnt_qposadr[sj])
                self._rod_slide_dof.append(self.model.jnt_dofadr[sj])
                self._rod_rot_qpos.append(self.model.jnt_qposadr[rj])
                self._rod_rot_dof.append(self.model.jnt_dofadr[rj])

        # Kick/reward: rotation joint qpos addrs per rod, guy body ids
        # Organised as list of (kick_dir, rot_qpos_adr, [guy_body_ids])
        self._kick_rods = []
        for player in ['y', 'b']:
            kick_dir = 1.0 if player == 'y' else -1.0
            for rod in RODS:
                rj = _jid(f"{player}{rod}rotation")
                rot_qpos = self.model.jnt_qposadr[rj]
                guys = []
                for g in range(1, 6):
                    gid = _bid(f"{player}{rod}guy{g}")
                    if gid >= 0:
                        guys.append(gid)
                self._kick_rods.append((kick_dir, rot_qpos, guys))

        # Protagonist (yellow) guy body ids only — for contact reward
        self._yellow_guy_bids = []
        for rod in RODS:
            for g in range(1, 6):
                gid = _bid(f"y{rod}guy{g}")
                if gid >= 0:
                    self._yellow_guy_bids.append(gid)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Ball body pos is (0, -4, 1.705), so qpos_y=4 → world Y≈0 (field centre).
        xy_random = np.random.normal(
            loc=[0.0, 4.0],
            scale=[0.5, 0.5]
        )

        self.data.qpos[self._bx_qpos] = xy_random[0]
        self.data.qpos[self._by_qpos] = xy_random[1]

        self.simulation_time = 0
        self._step_count = 0
        self.prev_ball_y = self.data.qpos[self._by_qpos]
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

            # Scale from [-1,1] to same range as protagonist (±20) so both
            # teams have equal control authority, then mirror directions.
            antagonist_action = antagonist_action * 20.0
            antagonist_action = self._adjust_antagonist_action(antagonist_action)
        else:
            antagonist_action = np.zeros(self.antagonist_action_size)

        self.data.ctrl[:self.protagonist_action_size] = protagonist_action
        self.data.ctrl[self.protagonist_action_size:self.protagonist_action_size + self.antagonist_action_size] = antagonist_action

        mujoco.mj_step(self.model, self.data)
        self._apply_virtual_kicks()   # inject ball impulse if foosman near + rotated
        self.simulation_time += self.model.opt.timestep
        self._step_count += 1

        obs = self._get_obs()

        reward = self.compute_reward(protagonist_action)
        terminated = self.terminated

        info = {}

        return obs, reward, terminated, False, info

    def _get_ball_obs(self):
        # Uses cached addresses — zero mj_name2id calls.
        ball_pos = [
            self.data.qpos[self._bx_qpos],
            self.data.qpos[self._by_qpos],
            0.0
        ]
        ball_vel = [
            self.data.qvel[self._bx_dof],
            self.data.qvel[self._by_dof],
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

        # Uses cached addresses — zero mj_name2id calls
        for i in range(len(self._rod_slide_qpos)):
            rod_slide_positions.append(self.data.qpos[self._rod_slide_qpos[i]])
            rod_slide_velocities.append(self.data.qvel[self._rod_slide_dof[i]])
            rod_rotate_positions.append(self.data.qpos[self._rod_rot_qpos[i]])
            rod_rotate_velocities.append(self.data.qvel[self._rod_rot_dof[i]])

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

        ball_xy = self.data.body(self._ball_bid).xpos[:2]

        for kick_dir, rot_qpos, guys in self._kick_rods:
            rot_angle = self.data.qpos[rot_qpos]
            if abs(rot_angle) < KICK_MIN_ROT:
                continue
            for gid in guys:
                guy_xy = self.data.body(gid).xpos[:2]
                dist   = np.linalg.norm(ball_xy - guy_xy)
                if dist < KICK_RADIUS:
                    strength = min(abs(rot_angle), 1.5) / 1.5
                    impulse  = KICK_SPEED * strength * (1.0 - dist / KICK_RADIUS)
                    self.data.qvel[self._by_dof] += kick_dir * impulse
                    self.data.qvel[self._bx_dof] += (ball_xy[0] - guy_xy[0]) * 1.5
                    self._kick_cooldown = KICK_COOLDOWN
                    return

    def euclidean_goal_distance(self, x, y):
        return math.sqrt((x - 0) ** 2 + (y - TABLE_MAX_Y_DIM) ** 2)

    # ── Reward function ────────────────────────────────────────────────────────
    def compute_reward(self, protagonist_action):
        ball_pos, ball_vel = self._get_ball_obs()
        ball_speed = math.sqrt(ball_vel[0]**2 + ball_vel[1]**2)

        # Use world-frame Y for goal detection (ball body offset is -4)
        ball_world_y = self.data.body(self._ball_bid).xpos[1]

        # 1. GOALS — dominant signal ±3000
        victory = 3000.0 if ball_world_y >  TABLE_MAX_Y_DIM else 0.0
        loss    = -3000.0 if ball_world_y < -TABLE_MAX_Y_DIM else 0.0

        # Ball world-frame XY
        ball_world_xy = self.data.body(self._ball_bid).xpos[:2]

        # 2. CONTACT PROXIMITY — uses cached yellow guy body ids
        contact_reward = 0.0
        for gid in self._yellow_guy_bids:
            dx = ball_world_xy[0] - self.data.body(gid).xpos[0]
            dy = ball_world_xy[1] - self.data.body(gid).xpos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 8.0:
                contact_reward += 1.0 / (dist + 1.0)
        if contact_reward > 1.5:
            contact_reward = 1.5

        # 3. BALL POSITION — continuous gradient toward opponent goal
        position_reward = self.data.body(self._ball_bid).xpos[1] * 0.02

        # 4. FORWARD VELOCITY — reward ball moving toward +Y
        forward_reward = max(ball_vel[1], 0.0) * 0.5

        # 5. PENALTIES
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

        over_max_steps = self._step_count >= MAX_EPISODE_STEPS

        unhealthy = not self.is_healthy
        no_progress = self.no_progress_steps >= self.max_no_progress_steps

        # Use world-frame Y for goal detection
        ball_world_y = self.data.body(self._ball_bid).xpos[1]
        victory = ball_world_y < -TABLE_MAX_Y_DIM or ball_world_y > TABLE_MAX_Y_DIM

        terminated = (
                unhealthy or (no_progress and not self.play_until_goal)
                or ball_stagnant or over_max_steps or victory
        ) if self._terminate_when_unhealthy else False

        return terminated