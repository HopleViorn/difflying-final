import torch
import numpy as np
import matplotlib.pyplot as plt
from flyinglib.control.quad_lee_controller import QuadLeeController
from flyinglib.objects.drone import Drone
from flyinglib.simulation.step import StepLayer

def run_simulation():
    """运行模拟，展示使用基于Lee控制器的加速度控制"""
    # 设置参数
    batch_size = 1  # 单个无人机
    sim_steps = 500  # 模拟步数
    sim_dt = 0.02   # 模拟时间步长
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Running simulation on device: {device}")
    
    # 初始化无人机
    drone = Drone(
        'accel_controlled_drone',
        batch_size=batch_size,
        sim_steps=sim_steps,
        sim_dt=sim_dt,
        size=0.2,  # 机臂长度
        requires_grad=False  # 不需要计算梯度
    )

    # 初始化Lee控制器
    controller = QuadLeeController(num_envs=batch_size, device=device)
    
    # 初始状态
    init_pos = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)  # 初始位置
    init_vel = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)     # 初始速度
    init_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)  # 初始姿态，格式(x,y,z,w)
    init_ang_vel = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)      # 初始角速度
    
    # 初始化状态向量
    q = torch.cat([init_pos, init_quat], dim=1)
    qd = torch.cat([init_ang_vel, init_vel], dim=1)
    
    # 保存轨迹数据
    positions = [init_pos.clone().cpu().numpy()]
    orientations = [init_quat.clone().cpu().numpy()]
    velocities = [init_vel.clone().cpu().numpy()]
    angular_velocities = [init_ang_vel.clone().cpu().numpy()]
    motor_thrusts = []
    
    # 定义目标轨迹
    def get_target_position(t):
        """根据时间生成目标位置，这里使用一个螺旋上升轨迹"""
        radius = 0.5  # 减小轨迹半径，避免不稳定
        height_rate = 0.2  # 减慢上升速度
        angular_velocity = 0.3  # 减慢旋转速度
        
        # 将输入转换为PyTorch张量（如果还不是）
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device, dtype=torch.float32)
            
        x = radius * torch.cos(angular_velocity * t)
        # Y轴是向下的，所以我们取负值来让无人机上升
        y = -height_rate * t
        z = radius * torch.sin(angular_velocity * t)
        
        # 限制高度
        if isinstance(y, torch.Tensor):
            y = torch.clamp(y, min=-2.0)
        else:
            y = max(y, -2.0)
            
        return torch.tensor([[x.item(), y.item(), z.item()]], device=device, dtype=torch.float32)
    
    # 模拟循环
    print("Starting simulation...")
    had_nan = False
    for step in range(sim_steps):
        # 当前时间
        t = step * sim_dt
        
        # 获取当前状态
        current_pos = q[:, :3]
        current_quat = q[:, 3:7]
        current_ang_vel = qd[:, :3]
        current_vel = qd[:, 3:6]
        
        # 检查NaN值
        if torch.isnan(current_pos).any() or torch.isnan(current_quat).any():
            print(f"NaN值出现在步骤 {step}，停止模拟")
            had_nan = True
            break
        
        # 获取目标状态
        target_pos = get_target_position(t)
        
        # 计算目标速度（简单前向差分）
        if step > 0:
            prev_target_pos = get_target_position(t - sim_dt)
            target_vel = (target_pos - prev_target_pos) / sim_dt
        else:
            target_vel = torch.zeros_like(target_pos, dtype=torch.float32)
        
        # 使用控制器计算电机推力
        motor_actions = controller.compute_control_from_position_target(
            current_pos=current_pos,
            current_vel=current_vel,
            current_orientation=current_quat,
            current_ang_vel=current_ang_vel,
            target_pos=target_pos,
            target_vel=target_vel
        )
        
        # 执行模拟步骤
        q, qd = StepLayer.apply(q, qd, motor_actions, drone)
        
        # 保存轨迹数据
        positions.append(q[:, :3].clone().cpu().numpy())
        orientations.append(q[:, 3:7].clone().cpu().numpy())
        velocities.append(qd[:, 3:6].clone().cpu().numpy())
        angular_velocities.append(qd[:, :3].clone().cpu().numpy())
        motor_thrusts.append(motor_actions.clone().cpu().numpy())
        
        # 输出进度
        if step % 10 == 0:
            print(f"Step {step}/{sim_steps}, Current position: {current_pos.cpu().numpy()}")
            print(f"  Target position: {target_pos.cpu().numpy()}")
            print(f"  Motor thrusts: {motor_actions[0].cpu().numpy()}")
    
    # 转换为numpy数组以便绘图
    positions = np.array(positions)
    orientations = np.array(orientations)
    velocities = np.array(velocities)
    angular_velocities = np.array(orientations)
    motor_thrusts = np.array(motor_thrusts)
    
    # 如果出现了NaN，就截断数组
    if had_nan:
        valid_steps = len(positions)
        time_points = np.arange(0, valid_steps) * sim_dt
    else:
        time_points = np.arange(0, sim_steps + 1) * sim_dt
    
    # 生成目标轨迹进行对比
    target_positions = np.zeros((len(time_points), 3))
    for i, t in enumerate(time_points):
        target_positions[i] = get_target_position(t).cpu().numpy()[0]  # 注意这里提取[0]
    
    # 绘制结果
    plot_results(time_points, positions, target_positions, motor_thrusts)
    
    print("Simulation complete!")

def plot_results(time_points, actual_positions, target_positions, motor_thrusts):
    """绘制模拟结果"""
    plt.figure(figsize=(16, 12))
    
    # 压缩维度 - 移除批次维度
    if actual_positions.ndim > 2:
        actual_positions = actual_positions.reshape(-1, 3)
    
    # 绘制3D轨迹
    ax1 = plt.subplot(2, 2, 1, projection='3d')
    ax1.plot(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], 'b-', label='Actual')
    ax1.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 'r--', label='Target')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Drone Trajectory')
    ax1.legend()
    
    # 绘制位置误差
    ax2 = plt.subplot(2, 2, 2)
    position_error = np.sqrt(np.sum((actual_positions - target_positions)**2, axis=1))
    ax2.plot(time_points, position_error)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Position Error')
    ax2.grid(True)
    
    # 绘制电机推力
    ax3 = plt.subplot(2, 2, 3)
    for i in range(4):
        # 移除批次维度
        thrusts = motor_thrusts[:, 0, i] if motor_thrusts.ndim > 2 else motor_thrusts[:, i]
        ax3.plot(time_points[:-1], thrusts, label=f'Motor {i+1}')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Motor Thrust (normalized)')
    ax3.set_title('Motor Thrusts')
    ax3.legend()
    ax3.grid(True)
    
    # 绘制高度和XY位置
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(time_points, actual_positions[:, 0], 'r-', label='X')
    ax4.plot(time_points, actual_positions[:, 1], 'g-', label='Y (Height)')
    ax4.plot(time_points, actual_positions[:, 2], 'b-', label='Z')
    ax4.plot(time_points, target_positions[:, 0], 'r--', alpha=0.5)
    ax4.plot(time_points, target_positions[:, 1], 'g--', alpha=0.5)
    ax4.plot(time_points, target_positions[:, 2], 'b--', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('Position Components')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('acceleration_control_results.png')
    plt.show()

if __name__ == "__main__":
    run_simulation() 