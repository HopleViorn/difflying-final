import torch
import numpy as np
import math
from flyinglib.simulation.simulator import QuadSimulator

class AttitudeControlTask:
    """姿态控制任务：让无人机学会控制自身的姿态（俯仰、横滚、偏航）"""
    
    def __init__(self, sim, task_config):
        self.sim: QuadSimulator = sim
        self.task_config = task_config
        self.device = task_config.device
        self.num_envs = task_config.num_envs
        self.episode_len_steps = task_config.sim_steps
        
        # 设置控制模式 - 姿态控制一般需要低级控制
        self.sim.set_control_mode(high_level=False)
        
        # 初始化计数器
        self.reset_counters()
        
        # 目标高度和姿态
        self.hover_height = 0.5  # 悬停高度（米）
        self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_positions[:, 2] = self.hover_height
        
        # 初始化目标姿态（初始为水平姿态）
        self.initialize_target_attitudes()
        
        # 每个步骤更新目标姿态的概率
        self.attitude_change_prob = 0.005  # 低概率随机变化目标姿态
        
        # 用于计算奖励的变量
        self.prev_attitude_error = torch.zeros(self.num_envs, device=self.device)
        self.current_phase = torch.zeros(self.num_envs, device=self.device)  # 当前训练阶段
        
        # 存储任务特定的中间状态信息（用于reward计算）
        self.attitude_error = torch.zeros(self.num_envs, device=self.device)
        self.position_error = torch.zeros(self.num_envs, device=self.device)
        self.angular_velocity_error = torch.zeros(self.num_envs, device=self.device)
        self.velocity_magnitude = torch.zeros(self.num_envs, device=self.device)
        
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

    def initialize_target_attitudes(self):
        """初始化目标姿态"""
        # 初始化为水平姿态的四元数 (w, x, y, z) = (1, 0, 0, 0)
        self.target_quats = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_quats[:, 0] = 1.0  # w分量为1表示水平姿态
        
        # 目标角速度（初始为零）
        self.target_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)

    def euler_to_quat(self, roll, pitch, yaw):
        """欧拉角转换为四元数"""
        # 将角度转换为弧度
        roll_rad = roll
        pitch_rad = pitch
        yaw_rad = yaw
        
        # 计算各角的一半的正弦和余弦值
        cy = torch.cos(yaw_rad * 0.5)
        sy = torch.sin(yaw_rad * 0.5)
        cp = torch.cos(pitch_rad * 0.5)
        sp = torch.sin(pitch_rad * 0.5)
        cr = torch.cos(roll_rad * 0.5)
        sr = torch.sin(roll_rad * 0.5)
        
        # 四元数分量
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        
        # 返回形状为(batch_size, 4)的四元数
        return torch.stack([w, x, y, z], dim=1)

    def quat_error(self, q1, q2):
        """计算两个四元数之间的误差"""
        # 计算四元数的共轭(q2*)
        q2_conj = q2.clone()
        q2_conj[:, 1:] = -q2_conj[:, 1:]
        
        # 计算q1与q2*的乘积，表示从q2到q1的旋转
        # 四元数乘法: q1 * q2
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2_conj[:, 0], q2_conj[:, 1], q2_conj[:, 2], q2_conj[:, 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        # 构建误差四元数
        q_error = torch.stack([w, x, y, z], dim=1)
        
        # 提取轴角表示
        angle = 2 * torch.acos(torch.clamp(q_error[:, 0], -1.0, 1.0))
        
        return angle

    def update_target_attitudes(self):
        """根据一定概率更新目标姿态"""
        # 随机决定是否更新目标姿态
        update_mask = torch.rand(self.num_envs, device=self.device) < self.attitude_change_prob
        
        if not torch.any(update_mask):
            return
        
        # 获取需要更新的环境索引
        update_idxs = update_mask.nonzero(as_tuple=False).flatten()
        
        # 根据训练阶段生成目标姿态
        for idx in update_idxs:
            phase = self.current_phase[idx].item()
            
            if phase < 1.0:  # 阶段1: 仅偏航控制
                yaw = (torch.rand(1, device=self.device) * 2 - 1) * math.pi  # 随机偏航角[-π, π]
                roll = 0.0
                pitch = 0.0
            elif phase < 2.0:  # 阶段2: 偏航和俯仰控制
                yaw = (torch.rand(1, device=self.device) * 2 - 1) * math.pi
                pitch = (torch.rand(1, device=self.device) * 2 - 1) * 0.3  # 小角度俯仰[-0.3, 0.3]弧度
                roll = 0.0
            else:  # 阶段3: 全姿态控制
                yaw = (torch.rand(1, device=self.device) * 2 - 1) * math.pi
                pitch = (torch.rand(1, device=self.device) * 2 - 1) * 0.3
                roll = (torch.rand(1, device=self.device) * 2 - 1) * 0.3  # 小角度横滚[-0.3, 0.3]弧度
            
            # 将欧拉角转换为四元数
            target_quat = self.euler_to_quat(
                roll=torch.tensor([roll], device=self.device),
                pitch=torch.tensor([pitch], device=self.device),
                yaw=torch.tensor([yaw], device=self.device)
            )
            
            # 更新目标姿态
            self.target_quats[idx] = target_quat

    def reset(self):
        """重置环境"""
        # 重置记录变量
        self.reset_counters()
        
        # 重置四旋翼初始状态（通过模拟器）
        self.sim.reset_quads()
        
        # 重置目标姿态
        self.initialize_target_attitudes()
        
        # 初始化训练阶段（所有环境从阶段0开始）
        self.current_phase = torch.zeros(self.num_envs, device=self.device)
        
        # 初始化姿态误差
        quad_orientations = self.sim.get_quad_orientations()
        self.prev_attitude_error = self.quat_error(quad_orientations, self.target_quats)
        
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
        
        # 随机更新目标姿态
        self.update_target_attitudes()
        
        # 获取当前观察值和计算中间状态 - 必须在计算奖励前调用
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
            
            # 更新成功环境的训练阶段
            for idx in reset_idxs:
                if self.successes[idx]:
                    # 成功完成当前阶段，进入下一阶段
                    self.current_phase[idx] = min(self.current_phase[idx] + 0.25, 3.0)
        
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
            "consecutive_successes": self.consecutive_successes,
            "current_phase": self.current_phase
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
        
        # 计算到目标位置的向量和距离（用于保持悬停位置）
        vec_to_target = self.target_positions - quad_positions
        dist_to_target = torch.norm(vec_to_target, dim=1).unsqueeze(1)
        
        # 单位化目标向量（处理零距离情况）
        norm_vec_to_target = torch.zeros_like(vec_to_target)
        nonzero_indices = dist_to_target.squeeze() > 1e-6
        if nonzero_indices.any():
            norm_vec_to_target[nonzero_indices] = vec_to_target[nonzero_indices] / dist_to_target[nonzero_indices]
        
        # 构建观察向量 - 与NavigationTask完全一致的结构
        observations = torch.cat([
            quad_positions,                      # 3: 位置
            quad_velocities,                     # 3: 速度
            quad_orientations,                   # 4: 姿态四元数
            norm_vec_to_target,                  # 3: 归一化的目标向量
            dist_to_target,                      # 1: 到目标的距离
            quad_angular_velocities,             # 3: 角速度
        ], dim=1)
        
        # 计算额外信息并存储（用于奖励计算）
        # 姿态误差
        self.attitude_error = self.quat_error(quad_orientations, self.target_quats)
        # 角速度误差
        self.angular_velocity_error = torch.norm(quad_angular_velocities - self.target_ang_vel, dim=1)
        # 位置误差
        self.position_error = torch.norm(quad_positions - self.target_positions, dim=1)
        # 速度大小
        self.velocity_magnitude = torch.norm(quad_velocities, dim=1)
        
        return observations

    def compute_rewards_and_dones(self):
        """计算奖励和完成状态"""
        # 获取四旋翼状态
        quad_positions = self.sim.get_quad_positions()
        quad_orientations = self.sim.get_quad_orientations()
        
        # 使用存储的状态信息
        attitude_error = self.attitude_error
        position_error = self.position_error
        angular_velocity_error = self.angular_velocity_error
        velocity_magnitude = self.velocity_magnitude
        
        # 计算姿态误差变化（改善或恶化）
        attitude_improvement = self.prev_attitude_error - attitude_error
        self.prev_attitude_error = attitude_error
        
        # 根据训练阶段调整奖励权重
        phase_weights = torch.zeros((self.num_envs, 4), device=self.device)
        for i in range(self.num_envs):
            phase = self.current_phase[i].item()
            if phase < 1.0:  # 阶段1: 主要关注偏航控制和位置保持
                phase_weights[i] = torch.tensor([0.4, 0.3, 0.2, 0.1], device=self.device)
            elif phase < 2.0:  # 阶段2: 平衡姿态和位置
                phase_weights[i] = torch.tensor([0.3, 0.4, 0.2, 0.1], device=self.device)
            else:  # 阶段3: 更注重姿态控制
                phase_weights[i] = torch.tensor([0.2, 0.5, 0.2, 0.1], device=self.device)
        
        # 奖励组成部分
        
        # 1. 位置保持奖励
        position_reward = torch.exp(-5.0 * position_error)
        
        # 2. 姿态控制奖励
        max_attitude_error = torch.tensor(math.pi, device=self.device)
        attitude_reward = 1.0 - torch.clamp(attitude_error / max_attitude_error, 0.0, 1.0)
        
        # 3. 稳定性奖励（角速度误差和线速度都越小越好）
        stability_reward = torch.exp(-2.0 * (angular_velocity_error + velocity_magnitude))
        
        # 4. 姿态改善奖励
        improvement_reward = torch.clamp(attitude_improvement * 2.0, -0.2, 0.2)
        
        # 根据阶段权重组合奖励
        rewards = (phase_weights[:, 0] * position_reward + 
                  phase_weights[:, 1] * attitude_reward + 
                  phase_weights[:, 2] * stability_reward + 
                  phase_weights[:, 3] * improvement_reward)
        
        # 判断任务是否成功完成
        # 姿态误差阈值根据阶段调整
        attitude_threshold = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            phase = self.current_phase[i].item()
            if phase < 1.0:  # 阶段1: 偏航控制（允许较大误差）
                attitude_threshold[i] = 0.2
            elif phase < 2.0:  # 阶段2: 偏航+俯仰控制
                attitude_threshold[i] = 0.15
            else:  # 阶段3: 全姿态控制
                attitude_threshold[i] = 0.1
        
        position_threshold = 0.1  # 悬停位置误差阈值
        stability_threshold = 0.2  # 稳定性阈值
        
        is_stable = (attitude_error < attitude_threshold) & (position_error < position_threshold) & (angular_velocity_error < stability_threshold) & (velocity_magnitude < stability_threshold)
        
        # 更新连续成功计数
        self.consecutive_successes = torch.where(
            is_stable,
            self.consecutive_successes + 1,
            torch.zeros_like(self.consecutive_successes)
        )
        
        # 判断任务是否成功完成
        success_threshold = 20  # 连续成功步数阈值
        self.successes = self.consecutive_successes >= success_threshold
        
        # 判断是否结束回合
        dones = self.successes.clone()
        
        # 回合超时
        episode_timeout = self.progress_buf >= self.episode_len_steps
        dones = dones | episode_timeout
        
        # 无人机失控情况（例如：位置过远、高度过低）
        quad_height = quad_positions[:, 2]
        excessive_position_error = position_error > 1.0  # 位置误差过大
        crash = quad_height < 0.1  # 高度过低视为坠毁
        lost_control = excessive_position_error | crash
        
        dones = dones | lost_control
        
        return rewards, dones

    def reset_quads(self, env_ids):
        """重置指定环境ID的四旋翼"""
        # 设置随机初始位置（靠近目标位置但有一定随机性）
        num_resets = len(env_ids)
        
        reset_positions = torch.zeros((num_resets, 3), device=self.device)
        # xy位置在目标周围随机
        reset_positions[:, :2] = torch.rand((num_resets, 2), device=self.device) * 0.2 - 0.1
        # z位置（高度）在目标高度上下随机
        reset_positions[:, 2] = self.hover_height + torch.rand(num_resets, device=self.device) * 0.2 - 0.1
        
        # 随机初始姿态（但偏差不要太大）
        reset_orientations = torch.zeros((num_resets, 4), device=self.device)
        reset_orientations[:, 0] = 0.95 + torch.rand(num_resets, device=self.device) * 0.05  # w分量接近1
        
        # 归一化四元数
        quat_norm = torch.norm(reset_orientations, dim=1, keepdim=True)
        reset_orientations = reset_orientations / quat_norm
        
        # 重置速度为零或很小的随机值
        reset_velocities = (torch.rand((num_resets, 3), device=self.device) - 0.5) * 0.1
        reset_angular_velocities = (torch.rand((num_resets, 3), device=self.device) - 0.5) * 0.1
        
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
        
        # 重置姿态误差
        quad_orientations = self.sim.get_quad_orientations()
        self.prev_attitude_error[env_ids] = self.quat_error(
            quad_orientations[env_ids], self.target_quats[env_ids]
        )
        
        # 对于失败的环境，可能需要重新设置目标姿态
        failed_envs = env_ids[~self.successes[env_ids]]
        if len(failed_envs) > 0:
            # 对于失败的环境，设置更简单的目标姿态
            self.target_quats[failed_envs, 0] = 1.0  # 重置为水平姿态
            self.target_quats[failed_envs, 1:] = 0.0

    def log_training_stats(self):
        """记录训练统计信息"""
        if self.log_episode_count == 0:
            return
            
        steps_since_last_log = self.step_count - self.last_log_step
        avg_reward = self.log_rewards_sum / self.log_episode_count
        success_rate = self.log_successes_count / self.log_episode_count
        
        # 计算各阶段的环境数量
        phase_0 = torch.sum((self.current_phase < 1.0).float()).item()
        phase_1 = torch.sum(((self.current_phase >= 1.0) & (self.current_phase < 2.0)).float()).item()
        phase_2 = torch.sum((self.current_phase >= 2.0).float()).item()
        
        print(f"训练统计 - 步数: {self.step_count}, 回合: {self.episode_count}, "
              f"平均奖励: {avg_reward:.2f}, 成功率: {success_rate:.2f}, "
              f"阶段分布: [阶段1: {phase_0}/{self.num_envs}, 阶段2: {phase_1}/{self.num_envs}, 阶段3: {phase_2}/{self.num_envs}]") 