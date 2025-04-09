import torch
import numpy as np
import matplotlib.pyplot as plt
from flyinglib.control.quad_lee_controller import QuadLeeController
from flyinglib.objects.drone import Drone
from flyinglib.simulation.step import StepLayer

def run_direct_acceleration_control():
    """运行模拟，展示直接使用加速度控制四旋翼无人机
    
    坐标系：
    - X轴：前向
    - Y轴：向下（重力方向）
    - Z轴：右侧
    
    四元数格式：(x,y,z,w)
    """
    # 设置参数
    batch_size = 1  # 单个无人机
    sim_dt = 0.02   # 模拟时间步长
    sim_steps = int(5.0 * 1.0 / sim_dt)  # 模拟步数
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Running simulation on device: {device}")
    
    # 初始化无人机
    drone = Drone(
        'direct_accel_drone',
        batch_size=batch_size,
        sim_steps=sim_steps,
        sim_dt=sim_dt,
        requires_grad=False  # 不需要计算梯度
    )
    
    # 初始化Lee控制器
    controller = QuadLeeController(num_envs=batch_size, device=device, drone=drone)
    
    # 初始状态
    init_pos = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)  # 初始位置
    init_vel = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)     # 初始速度
    init_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)  # 初始姿态，格式(x,y,z,w)
    init_ang_vel = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)      # 初始角速度
    
    # 初始化状态向量
    q = torch.cat([init_pos, init_quat], dim=1)  # 位置和姿态
    qd = torch.cat([init_ang_vel, init_vel], dim=1)  # 角速度和线速度
    
    # 保存轨迹数据
    positions = [init_pos.clone().cpu().numpy()]
    orientations = [init_quat.clone().cpu().numpy()]
    velocities = [init_vel.clone().cpu().numpy()]
    angular_velocities = [init_ang_vel.clone().cpu().numpy()]
    motor_thrusts = []
    
    # 定义加速度命令生成函数
    def get_desired_acceleration(t, pos, vel, orientation, ang_vel):
        """
        根据当前状态和时间生成期望的加速度和角加速度
        
        这里演示几种不同的加速度命令模式:
        1. 0-2秒: 垂直起飞
        2. 2-5秒: 简单的悬停
        3. 5-10秒: 水平8字飞行
        4. 10+秒: 悬停并执行偏航旋转
        """
        # 初始化加速度和角加速度
        linear_accel = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)
        angular_accel = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)
        
        # 垂直起飞阶段
        if t < 2.0:
            # 垂直加速度，带阻尼以达到匀速
            target_height = -1.0
            height_error = target_height - pos[0, 1]  # Y轴是高度方向
            vertical_vel = vel[0, 1]
            
            # PD控制器计算垂直加速度 (Y轴正方向为上升)
            
            linear_accel[0, 1] = 5.0 * height_error - 2.0 * vertical_vel
            
        # 悬停阶段
        elif t < 3.0:
            # 保持当前高度
            target_height = 1.0
            height_error = target_height - pos[0, 1]  # Y轴是高度方向
            vertical_vel = vel[0, 1]
            
            # 位置保持的PD控制
            linear_accel[0, 0] = 3.0 * pos[0, 0] - 2.0 * vel[0, 0]  # 回到原点
            linear_accel[0, 1] = 5.0 * height_error - 2.0 * vertical_vel  # 保持高度
            linear_accel[0, 2] = 3.0 * pos[0, 2] - 2.0 * vel[0, 2]  # 回到原点

        elif t < 5.0:
            linear_accel[0, 0] = 0.0
            linear_accel[0, 1] = 0.0
            linear_accel[0, 2] = 0.0
            #尝试翻转
            angular_accel[0, 1] = 0.0
            angular_accel[0, 2] = 0.1
            angular_accel[0, 0] = 0.0
            
        # 水平前进
        elif t < 10.0:
            # 使无人机沿X轴做简单的正弦运动，同时保持高度
            target_height = 1.0
            height_error = target_height - pos[0, 1]
            vertical_vel = vel[0, 1]
            
            # 高度控制
            linear_accel[0, 1] = 5.0 * height_error - 2.0 * vertical_vel
            
            # X轴上的正弦运动
            freq = 0.5  # 频率
            amplitude = 0.8  # 振幅
            phase = (t - 3.0) * freq * 2 * np.pi  # 相位
            
            # 计算目标位置和速度
            target_x = amplitude * np.sin(phase)
            target_vx = amplitude * freq * 2 * np.pi * np.cos(phase)
            
            # 位置和速度误差
            pos_error_x = target_x - pos[0, 0]
            vel_error_x = target_vx - vel[0, 0]
            
            # X轴PD控制
            linear_accel[0, 0] = 3.0 * pos_error_x + 1.0 * vel_error_x
            
            # Z轴位置保持
            linear_accel[0, 2] = 3.0 * (0 - pos[0, 2]) - 2.0 * vel[0, 2]
        
        return linear_accel, angular_accel
    
    # 模拟循环
    print("Starting simulation...")
    for step in range(sim_steps):
        # 当前时间
        t = step * sim_dt
        
        # 获取当前状态
        current_pos = q[:, :3]
        current_quat = q[:, 3:7]
        current_ang_vel = qd[:, :3]
        current_vel = qd[:, 3:6]
        
        # 计算期望的加速度和角加速度
        desired_accel, desired_rot_accel = get_desired_acceleration(
            t, current_pos, current_vel, current_quat, current_ang_vel
        )
        
        # desired_accel = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)
        # desired_rot_accel = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)

        # 使用控制器将加速度转换为电机推力
        motor_actions = controller.accelerations_to_motor_thrusts(
            desired_accel, desired_rot_accel, current_quat
        )
        
        # 执行模拟步骤
        q, qd = StepLayer.apply(q, qd, motor_actions, drone)
        
        # 保存轨迹数据
        positions.append(q[:, :3].clone().cpu().numpy())
        orientations.append(q[:, 3:7].clone().cpu().numpy())
        velocities.append(qd[:, 3:6].clone().cpu().numpy())
        angular_velocities.append(qd[:, :3].clone().cpu().numpy())
        motor_thrusts.append(motor_actions.clone().cpu().numpy())
        drone.render([0,1.0,0], None)
        
        # 输出进度
        if step % 10 == 0:
            pos = q[:, :3].cpu().numpy()
            att = q[:, 3:7].cpu().numpy()
            vel = qd[:, 3:6].cpu().numpy()
            ang_vel = qd[:, :3].cpu().numpy()
            print(f"Step {step}/{sim_steps}, t={t:.2f}s")
            print(f"  Position (x,y,z): {pos[0]}")
            print(f"  Attitude (x,y,z,w): {att[0]}")
            print(f"  Velocity (x,y,z): {vel[0]}")
            print(f"  Angular Velocity (x,y,z): {ang_vel[0]}")
    # 转换数据为numpy数组以便绘图
    positions = np.array(positions)
    orientations = np.array(orientations)  # 这个变量在后面没有使用
    velocities = np.array(velocities)
    angular_velocities = np.array(angular_velocities)  # 确保这是angular_velocities而不是orientations
    motor_thrusts = np.array(motor_thrusts)
    time_points = np.arange(0, sim_steps + 1) * sim_dt

    
    
    # 检查数组长度是否一致
    min_length = min(len(positions), len(velocities), len(angular_velocities), len(motor_thrusts)+1)
    if min_length < len(time_points):
        time_points = time_points[:min_length]
        positions = positions[:min_length]
        velocities = velocities[:min_length]
        angular_velocities = angular_velocities[:min_length]
        if min_length > len(motor_thrusts):
            # motor_thrusts比其他数组少一个元素
            motor_thrusts = np.pad(motor_thrusts, ((0, 1), (0, 0), (0, 0)), mode='edge')

    drone.renderer.save()
    
    # 绘制结果
    plot_results(time_points, positions, velocities, angular_velocities, motor_thrusts)
    
    print("Simulation complete!")

def plot_results(time_points, positions, velocities, angular_velocities, motor_thrusts):
    """绘制模拟结果"""
    plt.figure(figsize=(16, 16))
    
    # 绘制3D轨迹
    ax1 = plt.subplot(3, 2, 1, projection='3d')
    ax1.plot(positions[:, 0, 0], positions[:, 0, 2], -positions[:, 0, 1], 'b-')  # 注意Y轴向下，所以绘图时取负
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_zlabel('Height (m)')
    ax1.set_title('Drone Trajectory')
    
    # 创建标记特定时间点的函数
    def mark_time_point(t_value, color):
        idx = int(t_value / (time_points[-1] / len(time_points)))
        if idx < len(positions):
            ax1.scatter(
                positions[idx, 0, 0], 
                positions[idx, 0, 2], 
                -positions[idx, 0, 1],  # 绘图时高度取负
                color=color, s=100, label=f't={t_value}s'
            )
    
    # 标记关键时间点
    mark_time_point(0, 'green')
    mark_time_point(2, 'red')
    mark_time_point(5, 'purple')
    mark_time_point(10, 'orange')
    ax1.legend()
    
    # 绘制位置分量
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(time_points, positions[:, 0, 0], 'r-', label='X')
    ax2.plot(time_points, positions[:, 0, 1], 'g-', label='Y')
    ax2.plot(time_points, positions[:, 0, 2], 'b-', label='Z')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position Components')
    ax2.legend()
    ax2.grid(True)
    
    # 绘制速度分量
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(time_points, velocities[:, 0, 0], 'r-', label='Vx')
    ax3.plot(time_points, velocities[:, 0, 1], 'g-', label='Vy')
    ax3.plot(time_points, velocities[:, 0, 2], 'b-', label='Vz')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity Components')
    ax3.legend()
    ax3.grid(True)
    
    # 绘制角速度分量
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(time_points, angular_velocities[:, 0, 0], 'r-', label='ωx')
    ax4.plot(time_points, angular_velocities[:, 0, 1], 'g-', label='ωy')
    ax4.plot(time_points, angular_velocities[:, 0, 2], 'b-', label='ωz')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angular Velocity (rad/s)')
    ax4.set_title('Angular Velocity Components')
    ax4.legend()
    ax4.grid(True)
    
    # 绘制电机推力
    ax5 = plt.subplot(3, 2, 5)
    for i in range(4):
        ax5.plot(time_points[1:], motor_thrusts[:, 0, i], label=f'Motor {i+1}')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Motor Thrust (normalized)')
    ax5.set_title('Motor Thrusts')
    ax5.legend()
    ax5.grid(True)
    
    # 绘制角速度分量（只使用已有的角速度数据）
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(time_points, angular_velocities[:, 0, 0], 'r-', label='ω_roll')
    ax6.plot(time_points, angular_velocities[:, 0, 1], 'g-', label='ω_yaw')
    ax6.plot(time_points, angular_velocities[:, 0, 2], 'b-', label='ω_pitch')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Angular Velocity (rad/s)')
    ax6.set_title('Angular Velocity')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('direct_acceleration_control_results.png')
    plt.show()

if __name__ == "__main__":
    run_direct_acceleration_control() 