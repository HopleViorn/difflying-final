import torch
import numpy as np
from flyinglib.simulation.simulator import QuadSimulator

class ForwardFlightTask:
    """前进飞行任务：让无人机学会沿着特定方向稳定前进"""
    
    def __init__(self, sim, task_config):
        self.sim: QuadSimulator = sim
        self.task_config = task_config
        self.device = task_config.device
        self.num_envs = task_config.num_envs
        self.episode_len_steps = task_config.sim_steps
        
        # 设置控制模式
        self.sim.set_control_mode(high_level=task_config.high_level_control)
        
        # 初始化计数器
        self.reset_counters()
        
        # 飞行通道设置（默认沿x轴正方向飞行）
        self.flight_direction = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.flight_height = 0.5  # 飞行高度 (米)
        self.flight_distance = 1.0  # 飞行距离目标 (米)
        
        # 设置起点和终点
        self.start_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.start_positions[:, 2] = self.flight_height  # 设置起点高度
        
        self.target_positions = self.start_positions.clone()
        self.target_positions[:, 0] = self.flight_distance  # 目标点在x轴正方向上
        
        # 用于计算奖励的变量
        self.prev_progress = torch.zeros(self.num_envs, device=self.device)
        
        # 存储额外状态信息的变量
        self.progress = torch.zeros(self.num_envs, device=self.device)
        self.normalized_progress = torch.zeros(self.num_envs, device=self.device)
        self.height_error = torch.zeros(self.num_envs, device=self.device)
        self.direction_alignment = torch.zeros(self.num_envs, device=self.device)
        self.stability = torch.zeros(self.num_envs, device=self.device)
        
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
        
        # 累积奖励变量，用于在中止时更新
        self.cumulative_rewards = torch.zeros(self.num_envs, device=self.device)
    
    def reset_counters(self):
        """重置计数器"""
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.consecutive_successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.total_successes = 0
        self.total_resets = 0

    def reset(self):
        """重置环境"""
        # 重置记录变量
        self.reset_counters()
        
        # 重置四旋翼初始状态（通过模拟器）
        self.sim.reset_quads()
        
        # 初始化进度指标（沿飞行方向的投影）
        quad_positions = self.sim.get_quad_positions()
        self.prev_progress = self.calculate_flight_progress(quad_positions)
        
        # 重置累积奖励
        self.cumulative_rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 获取初始观察值
        obs = self.get_observations()
        
        # 返回标准格式的观察
        info = {}
        
        # Gym API 要求返回 (obs, info) 格式
        return obs, info

    def calculate_flight_progress(self, positions):
        """计算飞行进度（沿飞行方向的投影距离）"""
        # 计算从起点沿飞行方向的投影距离
        vec_from_start = positions - self.start_positions
        progress = torch.sum(vec_from_start * self.flight_direction.unsqueeze(0), dim=1)
        return progress
    
    def transform_policy_action(self, actions):
        """
        将策略输出的动作转换为控制输入
        
        Args:
            actions: 形状为 (num_envs, 4) 的动作张量，范围在 [-1, 1] 之间
            
        Returns:
            转换后的控制输入
        """
        if self.task_config.high_level_control:
            # 高级控制: [vx, vy, vz, yaw_rate]
            max_velocity = torch.tensor([0.5, 0.3, 0.3], device=self.device)  # 限制最大速度
            max_yaw_rate = torch.tensor(0.5, device=self.device)  # 限制最大偏航角速率
            
            high_level_actions = torch.zeros_like(actions)
            high_level_actions[:, :3] = actions[:, :3] * max_velocity.unsqueeze(0)
            high_level_actions[:, 3] = actions[:, 3] * max_yaw_rate
            
            return high_level_actions
        else:
            # 低级控制: 将动作从 [-1, 1] 映射到 [0, 1] 范围的推力
            transformed_actions = (actions + 1.0) * 0.5
            transformed_actions = torch.clamp(transformed_actions, 0.0, 1.0)
            return transformed_actions
    
    def step(self, actions):
        """执行一步模拟"""
        # 变换动作
        processed_actions = self.transform_policy_action(actions)
        
        # 执行模拟步骤
        self.sim.step(processed_actions)

        # 更新计数器
        self.progress_buf += 1
        self.step_count += 1
        
        # 获取当前观察值（同时计算并更新额外状态信息）
        obs = self.get_observations()
        
        # 计算奖励和完成状态
        step_rewards, dones = self.compute_rewards_and_dones()
        
        # 累积奖励
        self.cumulative_rewards = self.cumulative_rewards + step_rewards
        
        # 处理环境重置
        reset_idxs = (self.reset_buf | dones).nonzero(as_tuple=False).flatten()
        
        # 准备返回的奖励，初始为零
        rewards = torch.zeros_like(step_rewards)
        
        if len(reset_idxs) > 0:
            # 对于中止的环境，返回累积奖励
            rewards[reset_idxs] = self.cumulative_rewards[reset_idxs]
            
            # 重置这些环境
            self.reset_quads(reset_idxs)
            self.episode_count += len(reset_idxs)
            self.log_episode_count += len(reset_idxs)
            
            # 记录本周期内成功的回合数
            successes_in_reset = self.successes[reset_idxs].sum().item()
            self.log_successes_count += successes_in_reset
        
        # 累积本周期内的奖励
        self.log_rewards_sum += rewards.sum().item()
        
        # 记录训练统计信息
        if self.step_count % self.log_interval == 0:
            self.log_training_stats()
            # 重置周期记录数据
            self.log_rewards_sum = 0
            self.log_episode_count = 0
            self.log_successes_count = 0
            self.last_log_step = self.step_count
        
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
        quad_orientations = self.sim.get_quad_orientations()
        quad_angular_velocities = self.sim.get_quad_angular_velocities()
        
        # 计算到目标的向量和距离
        vec_to_target = self.target_positions - quad_positions
        dist_to_target = torch.norm(vec_to_target, dim=1).unsqueeze(1)
        
        # 单位化目标向量（处理零距离情况）
        norm_vec_to_target = torch.zeros_like(vec_to_target)
        nonzero_indices = dist_to_target.squeeze() > 1e-6
        if nonzero_indices.any():
            norm_vec_to_target[nonzero_indices] = vec_to_target[nonzero_indices] / dist_to_target[nonzero_indices]
        
        # 构建观察向量 - 保持与NavigationTask完全一致的结构
        observations = torch.cat([
            quad_positions,                      # 3: 位置
            quad_velocities,                     # 3: 速度
            quad_orientations,                   # 4: 姿态四元数
            norm_vec_to_target,                  # 3: 归一化的目标向量
            dist_to_target,                      # 1: 到目标的距离
            quad_angular_velocities,             # 3: 角速度
        ], dim=1)
        
        # 计算并存储额外信息，但不作为观察的一部分返回
        # 这些信息可以在compute_rewards_and_dones中使用
        self.progress = self.calculate_flight_progress(quad_positions)
        self.normalized_progress = self.progress / self.flight_distance
        self.height_error = torch.abs(quad_positions[:, 2] - self.flight_height)
        
        # 计算前进方向与实际速度方向的一致性
        velocity_magnitude = torch.norm(quad_velocities, dim=1)
        self.direction_alignment = torch.zeros_like(velocity_magnitude)
        nonzero_velocity = velocity_magnitude > 1e-6
        if nonzero_velocity.any():
            velocity_direction = quad_velocities[nonzero_velocity] / velocity_magnitude[nonzero_velocity].unsqueeze(1)
            self.direction_alignment[nonzero_velocity] = torch.sum(
                velocity_direction * self.flight_direction.unsqueeze(0), 
                dim=1
            )
        
        # 存储姿态稳定性
        self.stability = quad_orientations[:, 0]  # w分量接近1表示姿态接近水平
        
        return observations
    
    def compute_rewards_and_dones(self):
        """计算奖励和完成状态"""
        # 获取四旋翼状态
        quad_positions = self.sim.get_quad_positions()
        quad_orientations = self.sim.get_quad_orientations()
        
        # 使用存储的值，确保在调用此方法前已调用get_observations
        progress = self.progress
        normalized_progress = self.normalized_progress
        height_error = self.height_error
        direction_alignment = self.direction_alignment
        stability = self.stability
        
        # 计算进度变化（飞行速度）
        progress_delta = progress - self.prev_progress
        self.prev_progress = progress
        
        # 计算横向偏差（垂直于飞行方向的偏差）
        vec_from_start = quad_positions - self.start_positions
        # 移除沿飞行方向的分量
        projection = torch.sum(vec_from_start * self.flight_direction.unsqueeze(0), dim=1, keepdim=True)
        projected_point = self.start_positions + projection * self.flight_direction.unsqueeze(0)
        lateral_error = torch.norm(quad_positions - projected_point, dim=1)
        
        # 奖励组成部分
        
        # 1. 前进奖励：鼓励沿目标方向快速前进
        forward_reward = torch.clamp(progress_delta * 5.0, -0.5, 0.5)  # 限制单步奖励范围
        
        # 2. 高度控制奖励：保持指定飞行高度
        height_reward = 0.2 * torch.exp(-5.0 * height_error)
        
        # 3. 轨迹控制奖励：保持在飞行通道内
        lateral_reward = 0.2 * torch.exp(-5.0 * lateral_error)
        
        # 4. 稳定性奖励：保持稳定的姿态
        stability_reward = 0.1 * (stability - 0.5) * 2.0  # 将w从[0.5,1]映射到[0,1]
        
        # 5. 方向一致性奖励：速度方向与目标方向一致
        alignment_reward = 0.2 * torch.clamp(direction_alignment, 0.0, 1.0)
        
        # 6. 目标达成奖励：完成飞行任务
        completion_reward = torch.zeros_like(progress)
        completion_reward = torch.where(normalized_progress >= 1.0, torch.ones_like(progress), completion_reward)
        
        # 组合奖励
        rewards = forward_reward + height_reward + lateral_reward + stability_reward + alignment_reward + completion_reward
        
        # 判断任务是否成功完成
        at_target = normalized_progress >= 1.0
        stable_flight = (height_error < 0.1) & (lateral_error < 0.1) & (stability > 0.8)
        
        # 更新连续成功计数
        self.consecutive_successes = torch.where(
            at_target & stable_flight,
            self.consecutive_successes + 1,
            torch.zeros_like(self.consecutive_successes)
        )
        
        # 判断任务是否成功完成（到达目标且飞行稳定）
        success_threshold = 5  # 连续成功步数阈值
        self.successes = self.consecutive_successes >= success_threshold
        
        # 判断是否结束回合
        dones = self.successes.clone()
        
        # 回合超时
        episode_timeout = self.progress_buf >= self.episode_len_steps
        dones = dones | episode_timeout
        
        # 无人机失控情况（例如：过度倾斜、高度过低、严重偏离轨道）
        quad_height = quad_positions[:, 2]
        excessive_tilt = stability < 0.4  # 倾斜角度过大
        crash = quad_height < 0.1  # 高度过低视为坠毁
        severe_deviation = lateral_error > 0.5  # 严重偏离飞行通道
        lost_control = excessive_tilt | crash | severe_deviation
        
        dones = dones | lost_control
        
        return rewards, dones
    
    def reset_quads(self, env_ids):
        """重置指定环境ID的四旋翼"""
        # 设置随机初始位置（靠近起点但有一定随机性）
        num_resets = len(env_ids)
        
        reset_positions = torch.zeros((num_resets, 3), device=self.device)
        # 沿飞行方向随机小偏差
        reset_positions[:, 0] = torch.rand(num_resets, device=self.device) * 0.1
        # 垂直于飞行方向随机小偏差
        reset_positions[:, 1] = torch.rand(num_resets, device=self.device) * 0.1 - 0.05
        # 高度随机小偏差
        reset_positions[:, 2] = self.flight_height + torch.rand(num_resets, device=self.device) * 0.1 - 0.05
        
        # 初始朝向为随机偏航角，但保持水平姿态
        reset_orientations = torch.zeros((num_resets, 4), device=self.device)
        reset_orientations[:, 0] = 1.0  # 四元数w分量，表示水平姿态
        
        # 重置速度为零或小的正向速度
        reset_velocities = torch.zeros((num_resets, 3), device=self.device)
        reset_velocities[:, 0] = torch.rand(num_resets, device=self.device) * 0.1  # 小的初始正向速度
        
        reset_angular_velocities = torch.zeros((num_resets, 3), device=self.device)
        
        # 调用模拟器的重置函数
        self.sim.reset_quad_states(
            env_ids, 
            reset_positions, 
            reset_orientations, 
            reset_velocities, 
            reset_angular_velocities
        )
        
        # 重置这些环境的进度和重置标志
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        
        # 重置累积奖励
        self.cumulative_rewards[env_ids] = 0
        
        # 重置飞行进度
        quad_positions = self.sim.get_quad_positions()
        self.prev_progress[env_ids] = self.calculate_flight_progress(quad_positions[env_ids])

    def log_training_stats(self):
        """记录训练统计信息"""
        if self.log_episode_count == 0:
            return
            
        steps_since_last_log = self.step_count - self.last_log_step
        avg_reward = self.log_rewards_sum / self.log_episode_count
        success_rate = self.log_successes_count / self.log_episode_count
        
        print(f"训练统计 - 步数: {self.step_count}, 回合: {self.episode_count}, "
              f"平均奖励: {avg_reward:.2f}, 成功率: {success_rate:.2f}")