import torch
import numpy as np
from flyinglib import FLYINGLIB_DIRECTORY
from flyinglib.simulation.simulator import QuadSimulator

class NavigationTask:
    """导航任务：让无人机从起点飞到指定目标点"""
    
    def __init__(self, sim, task_config):
        self.sim: QuadSimulator = sim
        self.task_config = task_config
        self.device = task_config.device
        self.num_envs = task_config.num_envs
        self.episode_len_steps = task_config.sim_steps
        
        # 设置为高级控制模式
        self.sim.set_control_mode(high_level=True)
        
        # 初始化计数器
        self.reset_counters()
        
        # 初始化目标位置和起始位置
        self.initialize_target_positions()
        
        # 用于计算奖励的变量
        self.prev_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        
        # 添加记录器
        self.rewards_sum = 0
        self.episode_count = 0
        self.step_count = 0
        self.log_interval = 300
        
        # 添加用于周期性记录的变量
        self.log_rewards_sum = 0
        self.log_episode_count = 0
        self.log_successes_count = 0
        self.last_log_step = 0
        self.log_steps_before_reset_sum = 0
        
        # 添加奖励分量的字典，用于累计各个奖励分量
        self.reward_components_sum = {
            "pos_reward": 0,
            "close_to_goal_reward": 0,
            "getting_closer_reward": 0,
            "collision_penalty": 0,
            "crash_penalty": 0,
            "attitude_reward": 0,
            "velocity_reward": 0,
            "thrust_balance_reward": 0,
            "action_diff_penalty": 0,
            "action_magnitude_reward": 0
        }
        
        # 添加用于记录重置位置的变量
        self.reset_positions_sum = torch.zeros(3, device=self.device)
        self.reset_count = 0
        
        # 添加累积奖励变量，用于在中止时更新
        self.cumulative_rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 添加动作转换函数，将策略输出转换为高级控制输入

    def reset_counters(self):
        """重置计数器"""
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.consecutive_successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.total_successes = 0
        self.total_resets = 0
        self.total_crashes = 0
    def initialize_target_positions(self):
        """初始化目标位置"""
        # 环境边界（假设已在模拟器中设置）
        self.env_bounds = torch.tensor([1.0, 1.0, 1.0], device=self.device)  # [x, y, z] 范围
        
        # 根据配置设置目标范围
        target_min = self.env_bounds * torch.tensor(self.task_config.target_min_ratio, device=self.device)
        target_max = self.env_bounds * torch.tensor(self.task_config.target_max_ratio, device=self.device)
        
        # 随机生成目标位置

        self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        # self.target_positions[:, 0] = torch.rand(self.num_envs, device=self.device) * (target_max[0] - target_min[0]) + target_min[0]
        self.target_positions[:, 1] = torch.rand(self.num_envs, device=self.device) * (target_max[1] - target_min[1]) + target_min[1]
        # self.target_positions[:, 2] = torch.rand(self.num_envs, device=self.device) * (target_max[2] - target_min[2]) + target_min[2]

    def reset(self):
        """重置环境"""
        # 重置记录变量
        self.reset_counters()
        
        # 重置四旋翼初始状态（通过模拟器）
        self.sim.reset_quads()
        
        # 生成新的目标
        self.initialize_target_positions()
        
        # 初始化先前距离
        quad_positions = self.sim.get_quad_positions()
        self.prev_dist_to_target = torch.norm(quad_positions - self.target_positions, dim=1)
        
        # 重置累积奖励
        self.cumulative_rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 获取初始观察值
        obs = self.get_observations()
        
        # 返回标准格式的观察
        info = {}
        
        # Gym API 要求返回 (obs, info) 格式
        return obs, info
        
    def step(self, actions):
        """执行一步模拟"""
        # 变换动作（如果需要）
        processed_actions = self.task_config.action_transformation_function(actions)

        self.sim.step(processed_actions)

        # 更新计数器
        self.progress_buf += 1
        self.step_count += 1
        
        # 计算奖励和完成状态
        step_rewards, dones = self.compute_rewards_and_dones()
        
        # 累积奖励
        self.cumulative_rewards = self.cumulative_rewards * self.task_config.discount + step_rewards
        
        # 获取当前观察值
        obs = self.get_observations()
        
        # 处理环境重置
        reset_idxs = (self.reset_buf | dones).nonzero(as_tuple=False).flatten()
        

        rewards = step_rewards
        
        # rewards = torch.zeros_like(step_rewards)
        if len(reset_idxs) > 0:
            # 对于中止的环境，返回累积奖励
            # rewards[reset_idxs] = self.cumulative_rewards[reset_idxs]
            
            # 记录重置前的位置
            quad_positions = self.sim.get_quad_positions()
            reset_positions = quad_positions[reset_idxs]
            self.reset_positions_sum += reset_positions.sum(dim=0)
            self.reset_count += len(reset_idxs)
            self.log_steps_before_reset_sum += self.progress_buf[reset_idxs].sum().item()

            self.reset_quads(reset_idxs)
            self.episode_count += len(reset_idxs)
            self.log_episode_count += len(reset_idxs)
            
            # 记录本周期内成功的回合数
            successes_in_reset = self.successes[reset_idxs].sum().item()
            self.log_successes_count += successes_in_reset
        
        # 累积总奖励
        # self.rewards_sum += rewards.sum().item()
        # 累积本周期内的奖励
        self.log_rewards_sum += rewards.sum().item()
        
        # 记录训练统计信息
        if self.step_count % self.log_interval == 0:
            self.log_training_stats(rewards)
            # 重置周期记录数据
            self.log_rewards_sum = 0
            self.log_episode_count = 0
            self.log_successes_count = 0
            self.last_log_step = self.step_count
            # 重置位置记录
            self.reset_positions_sum = torch.zeros(3, device=self.device)
            self.log_steps_before_reset_sum = 0
            self.reset_count = 0
            self.total_crashes = 0
            
            # 重置奖励分量字典
            for key in self.reward_components_sum:
                self.reward_components_sum[key] = 0
        
        # 构建信息字典
        info = {
            "successes": self.successes,
            "consecutive_successes": self.consecutive_successes
        }
        
        # 将 dones 拆分为 terminated 和 truncated
        episode_timeout = self.progress_buf >= self.episode_len_steps
        terminated = dones & ~episode_timeout  # 任务完成导致的终止
        truncated = dones & episode_timeout    # 时间限制导致的截断
        
        # 返回 Gym API 格式
        return obs, rewards, terminated, truncated, info

    def get_observations(self):
        """获取观察值"""
        # 获取四旋翼状态
        quad_positions = self.sim.get_quad_positions()
        quad_velocities = self.sim.get_quad_velocities()
        quad_orientations = self.sim.get_quad_orientations()  # 四元数
        quad_angular_velocities = self.sim.get_quad_angular_velocities()  # 添加角速度信息
        
        # 计算到目标的向量
        vec_to_target = self.target_positions - quad_positions
        dist_to_target = torch.norm(vec_to_target, dim=1).unsqueeze(1)
        
        # 单位化目标向量
        norm_vec_to_target = vec_to_target / dist_to_target
        
        # 构建观察向量
        observations = torch.cat([
            quad_positions,                      # 3
            quad_velocities,                     # 3
            quad_orientations,                   # 4
            norm_vec_to_target,                  # 3
            dist_to_target,                      # 1
            quad_angular_velocities,             # 3 (添加角速度信息)
        ], dim=1)
        
        # 返回统一格式的观察
        return observations

    def compute_rewards_and_dones(self):
        """计算奖励和完成状态"""
        # 获取四旋翼位置和上一步动作
        quad_positions = self.sim.get_quad_positions()
        last_actions = self.sim.get_last_actions()
        
        # 计算到目标的距离
        vec_to_target = self.target_positions - quad_positions
        dist_to_target = torch.norm(vec_to_target, dim=1)
        
        # 获取四旋翼速度
        quad_velocities = self.sim.get_quad_velocities()
        velocity_magnitude = torch.norm(quad_velocities, dim=1)

        # 奖励参数
        rp = self.task_config.reward_parameters
        
        # 计算位置奖励（离目标越近奖励越高）
        pos_reward = rp["pos_reward_magnitude"] * torch.exp(-dist_to_target ** rp["pos_reward_exponent"])
        
        # 计算非常接近目标时的额外奖励
        close_to_goal_reward = torch.zeros_like(pos_reward)
        very_close_mask = dist_to_target < 0.3  # 距离小于 0.5 米视为非常接近
        close_to_goal_reward[very_close_mask] = rp["very_close_to_goal_reward_magnitude"] * torch.exp(
            -dist_to_target[very_close_mask] ** rp["very_close_to_goal_reward_exponent"]
        )
        
        # 计算接近目标的进度奖励（越来越接近目标）
        getting_closer_reward = torch.zeros_like(pos_reward)
        closer_mask = dist_to_target < self.prev_dist_to_target
        getting_closer_reward[closer_mask] = rp["getting_closer_reward_multiplier"] * (
            self.prev_dist_to_target[closer_mask] - dist_to_target[closer_mask]
        )
        
        # 计算动作平滑度惩罚
        action_diff_penalty = torch.zeros_like(pos_reward)
        if hasattr(self.sim, 'get_action_diff'):
            action_diff = self.sim.get_action_diff()
            x_diff_penalty = -rp["x_action_diff_penalty_magnitude"] * (
                torch.abs(action_diff[:, 0]) ** rp["x_action_diff_penalty_exponent"]
            )
            z_diff_penalty = -rp["z_action_diff_penalty_magnitude"] * (
                torch.abs(action_diff[:, 2]) ** rp["z_action_diff_penalty_exponent"]
            )
            yaw_diff_penalty = -rp["yawrate_action_diff_penalty_magnitude"] * (
                torch.abs(action_diff[:, 3]) ** rp["yawrate_action_diff_penalty_exponent"]
            )
            action_diff_penalty = x_diff_penalty + z_diff_penalty + yaw_diff_penalty
        
        # 计算动作大小奖励 (动作幅度越小奖励越大)
        action_magnitude_reward = torch.zeros_like(pos_reward)
        if last_actions is not None:
            # 计算动作幅度大小
            action_magnitude = torch.norm(last_actions, dim=1)
            # 使用负指数函数，动作幅度越小，奖励越大
            action_magnitude_reward = rp.get("action_magnitude_reward", 0.3) * torch.exp(-action_magnitude ** 2 / (2 * 0.5 ** 2))
        
        # 碰撞惩罚
        collision_penalty = torch.zeros_like(pos_reward)
        if hasattr(self.sim, 'get_collision_status'):
            collision_mask = self.sim.get_collision_status()
            collision_penalty[collision_mask] = rp["collision_penalty"]
        
        # 坠机惩罚 (高度小于-0.5视为坠机)
        crash_penalty = torch.zeros_like(pos_reward)
        
        # 获取无人机的姿态（四元数）
        quad_orientations = self.sim.get_quad_orientations()
        
        # 将四元数转换为欧拉角（俯仰角和横滚角）
        x, y, z, w = quad_orientations.unbind(dim=1)
        
        # 计算俯仰角（pitch）和横滚角（roll）
        pitch = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        roll = torch.atan2(2 * (w * y - z * x), 1 - 2 * (y * y + z * z))
        
        # 将弧度转换为角度
        pitch_deg = torch.rad2deg(pitch)
        roll_deg = torch.rad2deg(roll)
        
        height_crash_mask = (quad_positions[:, 1] < -2) | (quad_positions[:, 1] > 2) | \
                         (quad_positions[:, 2] < -2) | (quad_positions[:, 2] > 2) | \
                         (quad_positions[:, 0] < -2) | (quad_positions[:, 0] > 2)
        flip_crash_mask = (torch.abs(pitch_deg) > 45) | (torch.abs(roll_deg) > 45)  # 俯仰角或横滚角超过45度
        crash_mask = height_crash_mask | flip_crash_mask
        
        crash_penalty[crash_mask] = rp["crash_penalty"]

        # 计算姿态奖励（鼓励稳定的姿态）
        attitude_reward = torch.zeros_like(pos_reward)
        
        # 将俯仰角和横滚角转换为弧度制的绝对值
        pitch_rad = torch.abs(pitch)
        roll_rad = torch.abs(roll)
        
        # 计算姿态奖励（使用负指数函数，姿态越接近水平，奖励越高）
        attitude_reward = rp.get("attitude_reward_magnitude", 0.3) * (
            torch.exp(-pitch_rad ** 2 / (2 * 0.3 ** 2)) * 
            torch.exp(-roll_rad ** 2 / (2 * 0.3 ** 2))
        )

        # 计算基于距离的速度奖励（距离目标越近，速度越小越好）
        velocity_reward = torch.zeros_like(pos_reward)
        close_to_target_mask = dist_to_target < 0.3  # 当距离小于1米时开始考虑速度
        
        # 在接近目标时，速度越小奖励越大
        velocity_reward[close_to_target_mask] = rp.get("velocity_reward_magnitude", 10) * (
            torch.exp(-velocity_magnitude[close_to_target_mask] ** 2 / (2 * 0.5 ** 2)) * 
            (1.0 - dist_to_target[close_to_target_mask])  # 距离越近，速度奖励权重越大
        )

        # 获取最后的动作（四个旋翼的推力）
        last_actions = self.sim.get_last_actions()
        
        # 计算推力均衡奖励
        thrust_balance_reward = torch.zeros_like(pos_reward)
        if last_actions is not None:
            # 计算每个旋翼推力与平均推力的差异
            mean_thrust = torch.mean(last_actions, dim=1, keepdim=True)
            thrust_diff = torch.abs(last_actions - mean_thrust)
            max_thrust_diff = torch.max(thrust_diff, dim=1)[0]
            
            # 使用负指数函数计算奖励，推力差异越小，奖励越大
            thrust_balance_reward = rp.get("thrust_balance_magnitude", 0.5) * torch.exp(-max_thrust_diff ** 2 / (2 * 0.1 ** 2))

        # 总奖励（添加推力均衡奖励）
        # rewards = pos_reward + close_to_goal_reward + getting_closer_reward + collision_penalty*0 + crash_penalty + attitude_reward + velocity_reward + thrust_balance_reward + action_diff_penalty + action_magnitude_reward
        rewards = pos_reward 
        
        # 累加各个奖励分量到字典中
        reward_components = {
            "pos_reward": pos_reward,
            "close_to_goal_reward": close_to_goal_reward,
            "getting_closer_reward": getting_closer_reward,
            "collision_penalty": collision_penalty*0,
            "crash_penalty": crash_penalty,
            "attitude_reward": attitude_reward,
            "velocity_reward": velocity_reward,
            "thrust_balance_reward": thrust_balance_reward,
            "action_diff_penalty": action_diff_penalty,
            "action_magnitude_reward": action_magnitude_reward
        }
        
        for name, component in reward_components.items():
            self.reward_components_sum[name] += component.sum().item()
        
        # 存储当前距离用于下一步计算
        self.prev_dist_to_target = dist_to_target.clone()
        
        # 计算完成状态
        successes = dist_to_target < 0.3  # 到达目标的阈值，距离小于 0.3 米视为成功
        episode_timeout = self.progress_buf >= self.episode_len_steps
        
        # 更新成功次数
        self.successes = successes
        self.consecutive_successes = torch.where(
            successes, self.consecutive_successes + 1, torch.zeros_like(self.consecutive_successes)
        )
        
        # 确定哪些环境需要重置
        self.reset_buf = torch.where(
            episode_timeout,  # 现在坠机也会触发重置
            torch.ones_like(self.reset_buf),
            torch.zeros_like(self.reset_buf)
        )
        
        # 统计成功和重置次数
        num_resets = self.reset_buf.sum().item()
        num_successes = successes.sum().item()
        num_crashes = crash_mask.sum().item()
        self.total_resets += num_resets
        self.total_successes += num_successes
        self.total_crashes += num_crashes
        
        return rewards, self.reset_buf

    def reset_quads(self, env_ids):
        """重置特定环境的四旋翼"""
        # 重置特定四旋翼的状态（通过模拟器）
        self.sim.reset_quads(env_ids)
        
        # 为重置的环境重新初始化目标
        self.target_positions[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (
            self.env_bounds[0] * self.task_config.target_max_ratio[0] - 
            self.env_bounds[0] * self.task_config.target_min_ratio[0]
        ) + self.env_bounds[0] * self.task_config.target_min_ratio[0]
        
        self.target_positions[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (
            self.env_bounds[1] * self.task_config.target_max_ratio[1] - 
            self.env_bounds[1] * self.task_config.target_min_ratio[1]
        ) + self.env_bounds[1] * self.task_config.target_min_ratio[1]
        
        self.target_positions[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (
            self.env_bounds[2] * self.task_config.target_max_ratio[2] - 
            self.env_bounds[2] * self.task_config.target_min_ratio[2]
        ) + self.env_bounds[2] * self.task_config.target_min_ratio[2]
        
        # 重置计数器
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        
        # 重新计算到目标的初始距离
        quad_positions = self.sim.get_quad_positions()
        self.prev_dist_to_target[env_ids] = torch.norm(quad_positions[env_ids] - self.target_positions[env_ids], dim=1)
        
        # 重置累积奖励
        self.cumulative_rewards[env_ids] = 0

    def log_training_stats(self, rewards):
        """记录训练统计信息"""
        # 计算当前周期内的成功率
        if self.log_episode_count == 0:
            success_rate = 0.0
            avg_steps_before_reset = 0.0
        else:
            success_rate = self.log_successes_count / self.log_episode_count
            avg_steps_before_reset = self.log_steps_before_reset_sum / self.log_episode_count
        
        # 计算当前周期内的平均奖励
        steps_since_last_log = self.step_count - self.last_log_step
        avg_reward = self.log_rewards_sum / max(1, steps_since_last_log)
        
        # 计算各个奖励分量的平均值
        avg_reward_components = {}
        for name, value in self.reward_components_sum.items():
            avg_reward_components[name] = value / max(1, steps_since_last_log)
        
        # 计算重置时的平均位置
        if self.reset_count > 0:
            avg_reset_position = self.reset_positions_sum / self.reset_count
        else:
            avg_reset_position = torch.zeros(3, device=self.device)
        
        # 输出统计信息
        print(f"steps: {self.step_count}, episode: {self.episode_count}")
        print(f"reward: {avg_reward:.4f}")
        print(f"reward_components:")
        for name, value in avg_reward_components.items():
            print(f"  {name}: {value:.4f}")
        print(f"success_rate: {success_rate:.4f} ({self.log_successes_count}/{self.log_episode_count})")
        print(f"crashes: {self.total_crashes}")
        print(f"avg_steps_before_reset: ({avg_steps_before_reset:.2f}/{self.episode_len_steps})")
        print(f"avg_reset_position: [{avg_reset_position[0]:.4f}, {avg_reset_position[1]:.4f}, {avg_reset_position[2]:.4f}]")
        print("--------------------------------------------------------") 