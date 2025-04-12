import torch
import numpy as np

from flyinglib.control.quad_lee_controller import QuadLeeController
from flyinglib.objects.drone import Drone
from flyinglib.simulation.step import StepLayer

class QuadSimulator:
    """四旋翼飞行器模拟器
    
    注意：这只是一个示例实现，实际使用时应该连接到 flyinglib 的实际模拟系统
    """
    
    def __init__(self, task_config):
        """初始化模拟器
        
        Args:
            task_config: 任务配置
        """
        self.task_config = task_config
        self.device = task_config.device
        self.num_envs = task_config.num_envs

        #print info
        print(f"num_envs: {self.num_envs}")
        print(f"device: {self.device}")
        
        self.positions = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=True)
        self.velocities = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=True)
        self.angular_velocities = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=True)

        # 0,0,0,1
        self.orientations = torch.tensor([[0., 0., 0., 1.]], device=self.device, requires_grad=True).repeat(self.num_envs, 1)
        
        # 上一步的动作和当前动作
        self.last_actions = torch.zeros((self.num_envs, 6), device=self.device, requires_grad=True)
        self.current_actions = torch.zeros((self.num_envs, 6), device=self.device, requires_grad=True)
        
        # 碰撞状态
        self.collision_status = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # 初始化环境边界
        self.env_bounds = torch.tensor([10.0, 10.0, 3.0], device=self.device)
        
        # 初始化Drone实例
        self.drone = Drone(
            'quad_sim',
            batch_size=self.num_envs,
            sim_steps=self.task_config.sim_steps,
            sim_dt=self.task_config.sim_dt
        )

        self.controller = QuadLeeController(num_envs=self.num_envs, device=self.device, drone=self.drone)
        
        # 设置时间步长
        self.dt = task_config.sim_dt
        
        # 是否使用高级控制接口
        self.use_high_level_control = True
    
    def step(self, actions):
        """执行一步模拟
        
        Args:
            actions: 形状为 (num_envs, 4) 的动作张量
                     如果use_high_level_control=True，则为[vx, vy, vz, yaw_rate]
                     否则为直接的旋翼控制信号
            
        Returns:
            更新后的状态 (positions, velocities, orientations)
        """
        # 保存当前动作
        self.last_actions = self.current_actions
        self.current_actions = actions

        if self.use_high_level_control:
            low_level_actions = self.controller.accelerations_to_motor_thrusts(
                actions[:, :3], actions[:, 3:], self.orientations
            )
        else:
            low_level_actions = actions

        
        # 准备输入状态
        q = torch.cat([
            self.positions,
            self.orientations
        ], dim=1)

        
        qd = torch.cat([
            self.angular_velocities,
            self.velocities,
        ], dim=1)
        
        # 执行一步模拟
        next_q, next_qd = StepLayer.apply(q, qd, low_level_actions, self.drone)
        
        # 更新状态
        self.positions = next_q[:, :3]
        self.orientations = next_q[:, 3:]
        self.velocities = next_qd[:, 3:]
        self.angular_velocities = next_qd[:, :3]

        # 检查碰撞
        # self._check_collisions()
        
        return self.positions, self.velocities, self.orientations

    
    def reset_quads(self, env_ids=None):
        """重置四旋翼状态
        
        Args:
            env_ids: 要重置的环境 ID 列表，如果为 None，则重置所有环境
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # print(f"Resetting {len(env_ids)} quad(s)")
        
        # 随机化起始位置
        # self.positions[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * 0.5 - 0.25
        # self.positions[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * 0.5 - 0.25
        # self.positions[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * 1

        self.positions[env_ids] = torch.zeros((len(env_ids), 3), device=self.device, requires_grad=True)

        
        # 重置速度、角速度和方向
        self.velocities[env_ids] = 0.0
        self.angular_velocities[env_ids] = 0.0
        self.orientations[env_ids, 3] = 1.0
        self.orientations[env_ids, 0:3] = 0.0
        
        # 重置动作和碰撞状态
        self.last_actions[env_ids] = 0.0
        self.current_actions[env_ids] = 0.0
        self.collision_status[env_ids] = False
        
        # 重置无人机模拟状态
        self.drone.reset()
    
    def reset_quad_states(self, env_ids, positions, orientations, velocities, angular_velocities):
        """重置指定环境ID的四旋翼状态
        
        Args:
            env_ids: 要重置的环境ID列表
            positions: 新的位置，形状为 (len(env_ids), 3)
            orientations: 新的方向四元数，形状为 (len(env_ids), 4)
            velocities: 新的速度，形状为 (len(env_ids), 3)
            angular_velocities: 新的角速度，形状为 (len(env_ids), 3)
        """
        if len(env_ids) == 0:
            return
            
        # 更新状态
        self.positions[env_ids] = positions
        self.orientations[env_ids] = orientations
        self.velocities[env_ids] = velocities
        self.angular_velocities[env_ids] = angular_velocities
        
        # 重置动作和碰撞状态
        self.last_actions[env_ids] = 0.0
        self.current_actions[env_ids] = 0.0
        self.collision_status[env_ids] = False
        
        # 注意：由于我们修改了部分状态，这里不重置整个drone实例
    
    def _check_collisions(self):
        """检查四旋翼是否与环境边界或障碍物碰撞"""
        # 简化的边界碰撞检测
        out_of_bounds_lower = self.positions < -self.env_bounds
        out_of_bounds_upper = self.positions > self.env_bounds
        
        out_of_bounds = (out_of_bounds_lower | out_of_bounds_upper).any(dim=1)
        
        # 更新碰撞状态
        self.collision_status = out_of_bounds
        
        # 处理碰撞
        collision_indices = out_of_bounds.nonzero(as_tuple=False).flatten()
        if len(collision_indices) > 0:
            # 将碰撞的四旋翼反弹回边界内
            for i in collision_indices:
                for j in range(3):
                    if self.positions[i, j] < -self.env_bounds[j]:
                        self.positions[i, j] = -self.env_bounds[j]
                        self.velocities[i, j] = 0.0  # 停止在碰撞方向的运动
                    elif self.positions[i, j] > self.env_bounds[j]:
                        self.positions[i, j] = self.env_bounds[j]
                        self.velocities[i, j] = 0.0  # 停止在碰撞方向的运动
    
    def get_quad_positions(self):
        """获取所有四旋翼的位置
        
        Returns:
            形状为 (num_envs, 3) 的位置张量
        """
        return self.positions
    
    def get_quad_velocities(self):
        """获取所有四旋翼的速度
        
        Returns:
            形状为 (num_envs, 3) 的速度张量
        """
        return self.velocities
    
    def get_quad_orientations(self):
        """获取所有四旋翼的方向（四元数）
        
        Returns:
            形状为 (num_envs, 4) 的四元数张量
        """
        return self.orientations
    
    def get_quad_angular_velocities(self):
        """获取所有四旋翼的角速度
        
        Returns:
            形状为 (num_envs, 3) 的角速度张量
        """
        return self.angular_velocities
    
    def get_last_actions(self):
        """获取上一步的动作
        
        Returns:
            形状为 (num_envs, action_dim) 的动作张量
        """
        return self.current_actions
    
    def get_action_diff(self):
        """获取当前动作与上一步动作的差异
        
        Returns:
            形状为 (num_envs, action_dim) 的差异张量
        """
        return self.current_actions - self.last_actions
    
    def get_collision_status(self):
        """获取碰撞状态
        
        Returns:
            形状为 (num_envs,) 的布尔张量，表示每个环境中的四旋翼是否发生碰撞
        """
        return self.collision_status
    
    def set_control_mode(self, high_level=True):
        """设置控制模式
        
        Args:
            high_level: 是否使用高级控制模式
        """
        self.use_high_level_control = high_level