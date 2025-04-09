import torch
import numpy as np
import math

class QuadLeeController:
    """
    基于Lee控制器的四旋翼无人机控制器
    
    这个控制器可以将期望的加速度(ax, ay, az)和旋转加速度转换为四个旋翼的推力输出
    
    坐标系：
    - x轴：前向
    - y轴：向下（重力方向）
    - z轴：右侧
    
    四元数格式：(x,y,z,w)
    """
    
    def __init__(self, num_envs=1, device="cuda:0", drone=None):
        """
        初始化控制器
        
        Args:
            num_envs: 环境数量（用于批处理）
            device: 计算设备
        """
        self.num_envs = num_envs
        self.device = device
        
        # 设置无人机参数
        self.mass = 0.56  # kg，无人机质量
        self.gravity = torch.tensor([0.0, -9.81, 0.0], device=device, dtype=torch.float32).repeat(num_envs, 1)
        self.arm_length = 0.2  # m，旋臂长度
        self.motor_directions = torch.tensor([1, -1, 1, -1], device=device, dtype=torch.float32)  # 电机旋转方向，交替排列

        self.max_torque = 0.01
        self.max_thrust = 0.109919
        
        # 从propeller获取参数计算扭矩系数

        if drone is not None:
            self.max_torque = drone.max_torque
            self.max_thrust = drone.max_thrust
            self.mass = drone.mass
            self.inertia = drone.inertia
        print('max_torque: ', self.max_torque)
        print('max_thrust: ', self.max_thrust)
        print('inertia: ', self.inertia)
        print('mass: ', self.mass)
        print('static_action: ', drone.static_action)

        self.torque_constant = self.max_torque / self.max_thrust
        self.inertia = torch.tensor(self.inertia, device=self.device, dtype=torch.float32)
        
        # 初始化分配矩阵
        self._setup_allocation_matrix()
        
        # PID控制增益
        self.K_pos = torch.tensor([3.0, 3.0, 3.0], device=device, dtype=torch.float32).repeat(num_envs, 1)  # 位置控制增益
        self.K_vel = torch.tensor([2.0, 2.0, 2.0], device=device, dtype=torch.float32).repeat(num_envs, 1)  # 速度控制增益
        self.K_rot = torch.tensor([1.0, 1.0, 0.5], device=device, dtype=torch.float32).repeat(num_envs, 1)  # 姿态控制增益
        self.K_angvel = torch.tensor([0.1, 0.1, 0.1], device=device, dtype=torch.float32).repeat(num_envs, 1)  # 角速度控制增益
        
    def _setup_allocation_matrix(self):
        """构建推力分配矩阵"""
        # 创建分配矩阵 - 将力和力矩转换为四个电机的推力
        # 行：总推力(y)、x方向力矩、y方向力矩、z方向力矩
        # 列：四个电机的推力
        allocation_matrix = torch.zeros((4, 4), device=self.device, dtype=torch.float32)
        
        # 所有电机都产生Y方向推力（反重力方向）
        allocation_matrix[0, :] = 1.0
        
        # X方向力矩分配 (电机1和3在x轴)
        allocation_matrix[1, 0] = -self.arm_length  # 电机0贡献负x力矩
        allocation_matrix[1, 2] = self.arm_length   # 电机2贡献正x力矩
        
        # Z方向力矩分配 (电机0和2在z轴)
        allocation_matrix[3, 1] = self.arm_length   # 电机1贡献正z力矩
        allocation_matrix[3, 3] = -self.arm_length  # 电机3贡献负z力矩
        
        # Y方向力矩分配 (基于电机旋转方向)
        allocation_matrix[2, :] = self.motor_directions * self.torque_constant
        
        # 保存分配矩阵及其逆矩阵
        self.allocation_matrix = allocation_matrix
        self.inv_allocation_matrix = torch.inverse(allocation_matrix)
        
    def accelerations_to_motor_thrusts(self, desired_accel, desired_rot_accel, current_orientation):
        """
        将期望加速度和旋转加速度转换为四个电机的推力
        
        Args:
            desired_accel: 期望的线性加速度，形状(num_envs, 3)
            desired_rot_accel: 期望的角加速度，形状(num_envs, 3)
            current_orientation: 当前姿态四元数，形状(num_envs, 4)，格式(x,y,z,w)
        
        Returns:
            normalized_thrusts: 归一化的推力值，范围[0,1]，形状(num_envs, 4)
        """
        batch_size = desired_accel.shape[0]

        # print("desired_accel: ", desired_accel)
        # print("desired_rot_accel: ", desired_rot_accel)
        # print("current_orientation: ", current_orientation)
        
        # 确保所有张量使用相同的数据类型
        desired_accel = desired_accel.to(dtype=torch.float32)
        desired_rot_accel = desired_rot_accel.to(dtype=torch.float32)
        current_orientation = current_orientation.to(dtype=torch.float32)
        
        # 1. 计算期望总推力（考虑重力抵消）
        # 转换为体坐标系
        R = self._quat_to_rotmat(current_orientation)
        body_desired_accel = torch.bmm(R.transpose(1, 2), (desired_accel - self.gravity).unsqueeze(2)).squeeze(2)
        # print("body_desired_accel: ", body_desired_accel)
        
        # 2. 计算期望力矩
        # 将NumPy数组转换为PyTorch张量并扩展为批处理大小
        desired_torque = torch.bmm(self.inertia.unsqueeze(0).repeat(batch_size, 1, 1), desired_rot_accel.unsqueeze(2)).squeeze(2)
        
        # 限制最大力矩
        torque_norm = torch.norm(desired_torque, dim=1, keepdim=True)
        scale = torch.ones_like(torque_norm)
        mask = torque_norm > self.max_torque
        scale[mask] = self.max_torque / torque_norm[mask]
        desired_torque = desired_torque * scale
        
        # 3. 构建力/力矩向量
        wrench = torch.zeros((batch_size, 4), device=self.device, dtype=torch.float32)
        wrench[:, 0] = body_desired_accel[:, 1] * self.mass  # Y方向力，单位为N
        wrench[:, 1] = desired_torque[:, 0]  # X轴力矩
        wrench[:, 2] = desired_torque[:, 1]  # Y轴力矩
        wrench[:, 3] = desired_torque[:, 2]  # Z轴力矩

        # print("wrench: ", wrench)
        
        # 4. 计算电机推力
        wrench_reshaped = wrench.unsqueeze(-1)
        motor_thrusts = torch.matmul(self.inv_allocation_matrix, wrench_reshaped)
        motor_thrusts = motor_thrusts.squeeze(-1)
        
        motor_thrusts = torch.clamp(motor_thrusts, min=0.0)
        normalized_thrusts = motor_thrusts / self.max_thrust
        # print("normalized_thrusts: ", normalized_thrusts)
        
        return normalized_thrusts
    
    def compute_control_from_position_target(self, 
                                           current_pos, 
                                           current_vel, 
                                           current_orientation, 
                                           current_ang_vel,
                                           target_pos, 
                                           target_vel=None, 
                                           target_accel=None,
                                           target_yaw=None):
        """
        基于位置目标计算控制输出
        
        Args:
            current_pos: 当前位置，形状(num_envs, 3)
            current_vel: 当前速度，形状(num_envs, 3)
            current_orientation: 当前姿态四元数，形状(num_envs, 4)，格式(x,y,z,w)
            current_ang_vel: 当前角速度，形状(num_envs, 3)
            target_pos: 目标位置，形状(num_envs, 3)
            target_vel: 目标速度，形状(num_envs, 3)，默认为零
            target_accel: 目标加速度，形状(num_envs, 3)，默认为零
            target_yaw: 目标偏航角，形状(num_envs,)，默认为当前偏航角
        
        Returns:
            normalized_thrusts: 归一化的推力值，范围[0,1]，形状(num_envs, 4)
        """
        batch_size = current_pos.shape[0]
        
        # 确保所有输入都使用相同的数据类型
        current_pos = current_pos.to(dtype=torch.float32)
        current_vel = current_vel.to(dtype=torch.float32)
        current_orientation = current_orientation.to(dtype=torch.float32)
        current_ang_vel = current_ang_vel.to(dtype=torch.float32)
        target_pos = target_pos.to(dtype=torch.float32)
        
        # 默认参数处理
        if target_vel is None:
            target_vel = torch.zeros_like(current_vel, dtype=torch.float32)
        else:
            target_vel = target_vel.to(dtype=torch.float32)
            
        if target_accel is None:
            target_accel = torch.zeros_like(current_pos, dtype=torch.float32)
        else:
            target_accel = target_accel.to(dtype=torch.float32)
            
        if target_yaw is None:
            # 从四元数中计算当前偏航角
            _, _, yaw = self._quat_to_euler(current_orientation)
            target_yaw = yaw
        else:
            target_yaw = target_yaw.to(dtype=torch.float32)
            
        # 1. 计算位置和速度误差
        pos_error = target_pos - current_pos
        vel_error = target_vel - current_vel
        
        # 2. 计算期望加速度 (PD控制器)
        desired_accel = (
            self.K_pos * pos_error + 
            self.K_vel * vel_error + 
            target_accel  # 前馈项
        )
        
        # 3. 计算期望姿态
        # 计算期望朝向 - 修改为y轴为推力方向
        y_body_desired = self._compute_desired_thrust_axis(desired_accel)
        x_body_desired = self._compute_desired_forward_axis(y_body_desired, target_yaw)
        z_body_desired = torch.cross(x_body_desired, y_body_desired, dim=1)
        
        # 归一化
        x_body_desired = self._normalize_vectors(x_body_desired)
        y_body_desired = self._normalize_vectors(y_body_desired)
        z_body_desired = self._normalize_vectors(z_body_desired)
        
        # 构建期望旋转矩阵
        R_desired = torch.zeros((batch_size, 3, 3), device=self.device, dtype=torch.float32)
        R_desired[:, :, 0] = x_body_desired
        R_desired[:, :, 1] = y_body_desired
        R_desired[:, :, 2] = z_body_desired
        
        # 计算当前旋转矩阵
        R_current = self._quat_to_rotmat(current_orientation)
        
        # 4. 计算姿态误差
        R_error = torch.bmm(R_desired, R_current.transpose(1, 2))
        rot_error = self._vee_map(0.5 * (R_error - R_error.transpose(1, 2)))
        
        # 5. 计算期望角速度和角速度误差
        # 简化：期望角速度为零，因此角速度误差就是当前角速度
        angvel_error = current_ang_vel
        
        # 6. 计算期望角加速度 (PD控制器)
        desired_rot_accel = -(self.K_rot * rot_error + self.K_angvel * angvel_error)
        
        # 7. 计算电机推力
        return self.accelerations_to_motor_thrusts(desired_accel, desired_rot_accel, current_orientation)
    
    def _quat_to_rotmat(self, q):
        """
        四元数转换为旋转矩阵
        
        Args:
            q: 四元数，形状(num_envs, 4)，格式(x,y,z,w)
        
        Returns:
            R: 旋转矩阵，形状(num_envs, 3, 3)
        """
        batch_size = q.shape[0]
        
        # 确保四元数使用Float32数据类型
        q = q.to(dtype=torch.float32)
        qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # 构建旋转矩阵
        R = torch.zeros((batch_size, 3, 3), device=self.device, dtype=torch.float32)
        
        # 第一列
        R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
        R[:, 1, 0] = 2 * (qx * qy + qw * qz)
        R[:, 2, 0] = 2 * (qx * qz - qw * qy)
        
        # 第二列
        R[:, 0, 1] = 2 * (qx * qy - qw * qz)
        R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
        R[:, 2, 1] = 2 * (qy * qz + qw * qx)
        
        # 第三列
        R[:, 0, 2] = 2 * (qx * qz + qw * qy)
        R[:, 1, 2] = 2 * (qy * qz - qw * qx)
        R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
        
        return R
    
    def _quat_to_euler(self, q):
        """
        四元数转换为欧拉角 (ZYX顺序)
        
        Args:
            q: 四元数，形状(num_envs, 4)，格式(x,y,z,w)
        
        Returns:
            roll, pitch, yaw: 欧拉角，每个形状都是(num_envs,)
        """
        # 确保四元数使用Float32数据类型
        q = q.to(dtype=torch.float32)
        qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * torch.tensor(math.pi/2, device=self.device, dtype=torch.float32),
            torch.asin(sinp)
        )
        
        # yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def _compute_desired_thrust_axis(self, desired_accel):
        """
        计算期望的推力轴方向（y轴）
        
        Args:
            desired_accel: 期望的加速度，形状(num_envs, 3)
        
        Returns:
            y_body: 期望的推力方向，形状(num_envs, 3)
        """
        # 确保期望加速度使用Float32数据类型
        desired_accel = desired_accel.to(dtype=torch.float32)
        
        # 添加重力补偿
        accel_with_gravity = desired_accel - self.gravity
        
        # 归一化
        return self._normalize_vectors(accel_with_gravity)
    
    def _compute_desired_forward_axis(self, thrust_axis, target_yaw):
        """
        计算期望的前向轴方向（x轴）
        
        Args:
            thrust_axis: 期望的推力轴方向，形状(num_envs, 3)
            target_yaw: 目标偏航角，形状(num_envs,)
        
        Returns:
            x_body: 期望的前向轴方向，形状(num_envs, 3)
        """
        batch_size = thrust_axis.shape[0]
        
        # 确保输入张量使用Float32数据类型
        thrust_axis = thrust_axis.to(dtype=torch.float32)
        if isinstance(target_yaw, torch.Tensor):
            target_yaw = target_yaw.to(dtype=torch.float32)
        
        # 构建世界坐标系中的参考向量 - 指向前方(x轴正方向)
        yaw_reference = torch.zeros((batch_size, 3), device=self.device, dtype=torch.float32)
        yaw_reference[:, 0] = torch.cos(target_yaw)
        yaw_reference[:, 2] = torch.sin(target_yaw)  # 在xz平面内旋转
        
        # 计算z_body = thrust_axis × yaw_reference（叉积）
        z_body = torch.cross(thrust_axis, yaw_reference, dim=1)
        z_body = self._normalize_vectors(z_body)
        
        # 计算x_body = z_body × thrust_axis
        x_body = torch.cross(z_body, thrust_axis, dim=1)
        x_body = self._normalize_vectors(x_body)
        
        return x_body
    
    def _normalize_vectors(self, vectors):
        """
        归一化向量
        
        Args:
            vectors: 需要归一化的向量，形状(num_envs, 3)
        
        Returns:
            normalized_vectors: 归一化后的向量，形状(num_envs, 3)
        """
        # 确保输入向量使用Float32数据类型
        vectors = vectors.to(dtype=torch.float32)
        
        norm = torch.norm(vectors, dim=1, keepdim=True)
        mask = (norm > 1e-6).float()
        return mask * vectors / (norm + (1 - mask))
    
    def _vee_map(self, skew_matrix):
        """
        提取反对称矩阵的向量表示
        
        Args:
            skew_matrix: 反对称矩阵，形状(num_envs, 3, 3)
        
        Returns:
            vector: 向量表示，形状(num_envs, 3)
        """
        # 确保输入矩阵使用Float32数据类型
        skew_matrix = skew_matrix.to(dtype=torch.float32)
        
        batch_size = skew_matrix.shape[0]
        vector = torch.zeros((batch_size, 3), device=self.device, dtype=torch.float32)
        
        vector[:, 0] = skew_matrix[:, 2, 1]  # (2,1)元素
        vector[:, 1] = skew_matrix[:, 0, 2]  # (0,2)元素
        vector[:, 2] = skew_matrix[:, 1, 0]  # (1,0)元素
        
        return vector 
