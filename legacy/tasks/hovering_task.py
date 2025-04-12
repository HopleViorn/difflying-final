import torch
import numpy as np
from flyinglib.simulation.simulator import QuadSimulator

class HoveringTask:
    """悬停任务：让无人机学会在特定位置保持稳定悬停"""
    
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
        
        # 悬停目标高度（初始化为固定高度）
        self.target_height = torch.ones(self.num_envs, device=self.device) * 0.5  # 默认0.5米高
        self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_positions[:, 2] = self.target_height
        
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
        
        # 获取四旋翼位置并计算到目标的距离
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

    def transform_policy_action(self, actions):
        """
        将策略输出的动作转换为控制输入
        
        Args:
            actions: 形状为 (num_envs, 4) 的动作张量，范围在 [-1, 1] 之间
            
        Returns:
            转换后的控制输入
        """
        self.task_config.high_level_control = False
        if self.task_config.high_level_control:
            # 高级控制: [vx, vy, vz, yaw_rate]
            max_velocity = torch.tensor([0.2, 0.2, 0.2], device=self.device)  # 限制最大速度
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
        
        # 计算奖励和完成状态
        step_rewards, dones = self.compute_rewards_and_dones()
        
        # 累积奖励
        self.cumulative_rewards = self.cumulative_rewards + step_rewards
        
        # 获取当前观察值
        obs = self.get_observations()
        
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
        
        return observations

    def compute_rewards_and_dones(self):
        """计算奖励和完成状态"""
        # 获取四旋翼位置和姿态
        quad_positions = self.sim.get_quad_positions()
        quad_orientations = self.sim.get_quad_orientations()  # 四元数形式
        quad_velocities = self.sim.get_quad_velocities()
        quad_angular_velocities = self.sim.get_quad_angular_velocities()
        
        # 计算到目标的距离
        dist_to_target = torch.norm(quad_positions - self.target_positions, dim=1)
        
        # 提取四元数的标量部分(w)作为稳定性的度量
        stability = quad_orientations[:, 3]  # w分量接近1表示姿态接近水平
        
        # 计算速度和角速度的大小（稳定性的另一个度量）
        velocity_magnitude = torch.norm(quad_velocities, dim=1)
        angular_velocity_magnitude = torch.norm(quad_angular_velocities, dim=1)
        
        # 主要奖励：距离目标越近越好
        position_reward = 1.0 - torch.clamp(dist_to_target / 1.0, 0.0, 1.0)
        
        # 次要奖励：飞行器姿态越水平越好
        attitude_reward = (stability - 0.5) * 2.0  # 将w从[0.5,1]映射到[0,1]
        
        # 稳定性奖励：速度和角速度越小越好
        velocity_penalty = torch.clamp(velocity_magnitude / 1.0, 0.0, 1.0)
        angular_velocity_penalty = torch.clamp(angular_velocity_magnitude / 3.0, 0.0, 1.0)
        stability_reward = 1.0 - 0.5 * (velocity_penalty + angular_velocity_penalty)
        
        # 组合奖励
        rewards = 0.6 * position_reward + 0.2 * attitude_reward + 0.2 * stability_reward
        
        # 根据距离变化给予额外奖励/惩罚（鼓励接近目标）
        dist_improvement = self.prev_dist_to_target - dist_to_target
        rewards = rewards + 0.3 * torch.sign(dist_improvement)
        
        # 更新上一步距离
        self.prev_dist_to_target = dist_to_target
        
        # 任务成功条件：连续10步保持在目标附近稳定悬停
        hover_threshold = 0.1  # 悬停容差范围（米）
        stable_threshold = 0.2  # 稳定性容差（速度和角速度）
        
        # 检查悬停条件
        is_hovering = (dist_to_target < hover_threshold) & (velocity_magnitude < stable_threshold) & (angular_velocity_magnitude < stable_threshold)
        
        # 更新连续成功计数
        self.consecutive_successes = torch.where(
            is_hovering,
            self.consecutive_successes + 1,
            torch.zeros_like(self.consecutive_successes)
        )
        
        # 判断任务是否成功完成
        success_threshold = 10  # 连续成功步数阈值
        self.successes = self.consecutive_successes >= success_threshold
        
        # 判断是否结束回合
        dones = self.successes.clone()
        
        # 回合超时
        episode_timeout = self.progress_buf >= self.episode_len_steps
        dones = dones | episode_timeout
        
        # 无人机失控情况（例如：过度倾斜、高度过低）
        quad_height = quad_positions[:, 2]
        excessive_tilt = stability < 0.4  # w值过小表示倾斜角度过大
        crash = quad_height < 0.1  # 高度过低视为坠毁
        lost_control = excessive_tilt | crash
        
        dones = dones | lost_control
        
        return rewards, dones

    def reset_quads(self, env_ids):
        """重置指定环境ID的四旋翼"""
        # 设置随机初始位置（靠近目标但有一定偏差）
        num_resets = len(env_ids)
        
        reset_positions = torch.zeros((num_resets, 3), device=self.device)
        # xy位置在目标周围小范围随机
        reset_positions[:, :2] = torch.rand((num_resets, 2), device=self.device) * 0.2 - 0.1
        # z位置在目标高度上下小范围随机
        reset_positions[:, 2] = self.target_height[env_ids] + torch.rand(num_resets, device=self.device) * 0.2 - 0.1
        
        # 初始朝向为随机偏航角
        reset_orientations = torch.zeros((num_resets, 4), device=self.device)
        reset_orientations[:, 3] = 1.0  # 四元数w分量，表示水平姿态
        
        # 重置速度为零
        reset_velocities = torch.zeros((num_resets, 3), device=self.device)
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
        
        # 重置上一步距离
        quad_positions = self.sim.get_quad_positions()
        self.prev_dist_to_target[env_ids] = torch.norm(
            quad_positions[env_ids] - self.target_positions[env_ids], dim=1
        )

    def log_training_stats(self):
        """记录训练统计信息"""
        if self.log_episode_count == 0:
            return
            
        steps_since_last_log = self.step_count - self.last_log_step
        avg_reward = self.log_rewards_sum / self.log_episode_count
        success_rate = self.log_successes_count / self.log_episode_count
        
        print(f"训练统计 - 步数: {self.step_count}, 回合: {self.episode_count}, "
              f"平均奖励: {avg_reward:.2f}, 成功率: {success_rate:.2f}")