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
      2. Reduce ball joint friction from 20 → 2 (mass=0.1, so 20 gives 200 m/s² drag).
      3. Add damping to rotation joints to stop runaway spinning.
      4. Clamp rotation joint range to ±π.
    """
    # 1. Table-mesh collision off
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    if table_geom >= 0:
        model.geom_contype[table_geom] = 0
        model.geom_conaffinity[table_geom] = 0

    # 2. Reduce ball joint friction
    for jname in ["ball_x", "ball_y"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            dof = model.jnt_dofadr[jid]
            model.dof_frictionloss[dof] = 2.0

    # 3 & 4. Rotation joints: add damping, limit range
    for prefix in ["y_goal", "y_def", "y_mid", "y_attack",
                    "b_goal", "b_def", "b_mid", "b_attack"]:
        jnt_name = f"{prefix}_rotation"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jid >= 0:
            dof = model.jnt_dofadr[jid]
            model.dof_damping[dof] = 5000.0
            model.jnt_range[jid] = [-math.pi, math.pi]

from ai_agents.v2.gym.mujoco_table_render_mixin import MujocoTableRenderMixin

DIRECTION_CHANGE = 1
TABLE_MAX_Y_DIM = 65
BALL_STOPPED_COUNT_THRESHOLD = 300  # Was 50. Give it time to wind up!
MAX_STEPS = 2000                    # Was 1000. Let the games play out longer.

RODS = ["_goal_", "_def_", "_mid_", "_attack_"]

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
        self.max_no_progress_steps = 300  # Was 50. Give the agent time to wind up a strike.

        self.prev_ball_y = None
        self.no_progress_steps = 0
        self.ball_stopped_count = 0

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

        xy_random = np.random.normal(
            loc=[-0.5, 0.0],
            scale=[0.5, 0.5]
        )

        self.data.qpos[x_qpos_adr] = xy_random[0]
        self.data.qpos[y_qpos_adr] = xy_random[1]

        self.simulation_time = 0
        self.prev_ball_y = self.data.qpos[y_qpos_adr]
        self.no_progress_steps = 0
        self.ball_stopped_count = 0

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


    def euclidean_goal_distance(self, x, y):
        # Point (0, 64)
        return math.sqrt((x - 0) ** 2 + (y - TABLE_MAX_Y_DIM) ** 2)

    def compute_reward(self, protagonist_action):
        ball_obs = self._get_ball_obs()
        ball_pos = ball_obs[0]
        ball_vel = ball_obs[1]
        ball_y = ball_pos[1]
        ball_x = ball_pos[0]
        ball_z = ball_pos[2]
        
        # Calculate ball speed
        ball_speed = np.sqrt(ball_vel[0]**2 + ball_vel[1]**2)

        # PRIMARY REWARDS: Goals (dominant - keep these large)
        victory = 1000 * DIRECTION_CHANGE if ball_y >  TABLE_MAX_Y_DIM else 0  # Ball in antagonist's goal
        loss = -1000 * DIRECTION_CHANGE if ball_y < -1.0 * TABLE_MAX_Y_DIM else 0  # Ball in protagonist's goal

        # CURRICULUM LEARNING REWARDS: Help agent learn to interact with ball
        
        # 1. CONTACT REWARDS: Reward for getting foosmen near the ball
        contact_reward = 0.0
        ball_pos_2d = np.array([ball_x, ball_y, ball_z])
        
        # Check protagonist's foosmen (yellow team)
        for rod in RODS:
            rod_name = f"y{rod}rod"
            # Get positions of guys on this rod
            for guy_num in range(1, 6):  # Max 5 guys per rod
                guy_name = f"y{rod}guy{guy_num}"
                try:
                    guy_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, guy_name)
                    guy_pos = self.data.body(guy_body_id).xpos.copy()
                    distance = np.linalg.norm(ball_pos_2d - guy_pos)
                    
                    # Reward proximity, but keep it small enough that kicking is better
                    if distance < 10.0:
                        contact_reward += 5.0 / (distance + 1.0)  # Was 50.0
                    
                    # Small reward for close contact — not so large it discourages kicking
                    if distance < 3.0:
                        contact_reward += 10.0  # Was 200.0! This was the hugging trap.
                except:
                    continue  # Guy doesn't exist on this rod
        
        # 2. BALL MOVEMENT REWARDS: Buffed to encourage violent strikes
        movement_reward = 0.0
        if ball_speed > 0.1:
            movement_reward += ball_speed * 50.0  # Was 10.0. Massive multiplier for speed.
        
        # Extra reward for a solid strike
        if ball_speed > 1.0:                      # Was 0.5. Changed threshold to reward actual hits.
            movement_reward += 200.0              # Was 50.0. Big bonus for a hard kick.
        
        # 3. DIRECTIONAL REWARDS: Reward ball moving toward opponent goal - MASSIVELY INCREASED
        directional_reward = 0.0
        if ball_vel[1] > 0:  # Moving toward opponent (positive Y)
            directional_reward += ball_vel[1] * 25.0  # Was 5.0
            # Extra bonus if moving fast in right direction
            if ball_vel[1] > 2.0:
                directional_reward += 75.0  # Was 15.0
        
        # 4. POSITION REWARDS: Small rewards for ball position (only when moving)
        position_reward = 0
        if ball_speed > 0.5:
            position_reward = ball_y * 0.05  # Reduced from 0.1
        
        # 5. PENALTIES (reduced during learning phase)
        time_penalty = -1.0  # Reduced from -2.0 to be less harsh
        stagnant_penalty = -2.0 if ball_speed < 0.1 else 0  # Reduced from -5.0
        
        ctrl_cost = self.control_cost(protagonist_action)

        # Combine all rewards
        reward = (victory + loss + 
                 contact_reward + movement_reward + directional_reward + 
                 position_reward + time_penalty + stagnant_penalty - ctrl_cost)

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