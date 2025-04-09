# 四旋翼无人机加速度控制器

这个控制器实现了四旋翼无人机的加速度控制，能够将期望的加速度和角加速度命令转换为四个旋翼的推力输出。控制器基于Lee几何控制方法，在SO(3)上进行刚体控制。

## 坐标系定义

当前控制器使用以下坐标系：

- **X轴**：指向前方（无人机前进方向）
- **Y轴**：指向下方（重力方向，与重力平行）
- **Z轴**：指向右侧（遵循右手定则）

四元数格式为 **(x, y, z, w)**，与许多物理引擎的默认格式相同。

## 主要功能

控制器提供两种主要使用方式：

1. **直接加速度控制**：通过 `accelerations_to_motor_thrusts` 方法，直接将期望的线性加速度和角加速度转换为电机推力
2. **位置控制**：通过 `compute_control_from_position_target` 方法，提供目标位置、速度等，控制器会计算所需的加速度并输出电机推力

## 使用示例

### 1. 直接加速度控制

```python
# 初始化控制器
controller = QuadLeeController(num_envs=1, device="cuda:0")

# 定义期望的加速度和角加速度
# 注意：Y轴是重力方向，所以悬停时需要Y轴方向有-9.81的加速度（抵消重力）
desired_accel = torch.tensor([[0.0, -9.81, 0.0]], device=device)  # 悬停加速度命令
desired_rot_accel = torch.tensor([[0.0, 1.0, 0.0]], device=device)  # 绕Y轴旋转（偏航）

# 获取当前姿态四元数 - 格式是(x,y,z,w)
current_orientation = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)  # 水平姿态

# 计算电机推力
motor_thrusts = controller.accelerations_to_motor_thrusts(
    desired_accel, desired_rot_accel, current_orientation
)
```

### 2. 位置控制

```python
# 初始化控制器
controller = QuadLeeController(num_envs=1, device="cuda:0")

# 当前状态
current_pos = torch.tensor([[0.0, 0.0, 0.0]], device=device)  # (x,y,z)
current_vel = torch.tensor([[0.0, 0.0, 0.0]], device=device)  # (vx,vy,vz)
current_orientation = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)  # (x,y,z,w)
current_ang_vel = torch.tensor([[0.0, 0.0, 0.0]], device=device)  # (ωx,ωy,ωz)

# 目标位置 - 目标高度为1米（y值为1，因为y轴向下）
target_pos = torch.tensor([[1.0, 1.0, 1.0]], device=device)  # (x,y,z)

# 计算电机推力
motor_thrusts = controller.compute_control_from_position_target(
    current_pos=current_pos,
    current_vel=current_vel,
    current_orientation=current_orientation,
    current_ang_vel=current_ang_vel,
    target_pos=target_pos
)
```

## 坐标系注意事项

### 高度控制

由于Y轴是向下的：

- 在目标位置中，Y值越大表示越低
- 要让无人机上升，需要在Y方向施加负的加速度
- 重力加速度是正的Y方向(0, 9.81, 0)

### 旋转控制

- 偏航控制：主要通过Y轴角加速度控制
- 翻滚控制：主要通过X轴角加速度控制
- 俯仰控制：主要通过Z轴角加速度控制

## 控制原理

1. **推力分配矩阵**：将需要的总推力和三轴力矩映射到四个电机的推力输出

```
    # 所有电机都产生Y方向推力（重力方向）
    allocation_matrix[0, :] = 1.0
    
    # X方向力矩分配
    allocation_matrix[1, 0] = -arm_length  # 电机0贡献负x力矩
    allocation_matrix[1, 2] = arm_length   # 电机2贡献正x力矩
    
    # Z方向力矩分配
    allocation_matrix[3, 1] = arm_length   # 电机1贡献正z力矩
    allocation_matrix[3, 3] = -arm_length  # 电机3贡献负z力矩
    
    # Y方向力矩分配（基于电机旋转方向）
    allocation_matrix[2, :] = motor_directions * torque_constant
```

2. **期望加速度计算**：（在位置控制模式下）
```
desired_accel = K_pos * pos_error + K_vel * vel_error + target_accel
```

3. **期望姿态计算**：
   - 根据加速度计算期望推力方向（y轴）
   - 根据目标偏航角和推力方向计算期望前向方向（x轴）
   - 使用叉积计算期望右侧方向（z轴）

4. **姿态误差计算**：在SO(3)上计算当前姿态与期望姿态之间的误差

5. **电机推力计算**：使用推力分配矩阵的逆，将期望推力和力矩转换为各个电机的推力

## 实现加速度控制的常见任务

### 垂直起飞/降落
```python
# 垂直起飞 - 在Y轴施加-10m/s²加速度（抵消重力并上升）
accel = torch.tensor([[0.0, -15.0, 0.0]], device=device)
ang_accel = torch.zeros((1, 3), device=device)
```

### 水平飞行
```python
# 沿X方向飞行 - 保持高度并沿X轴加速
accel = torch.tensor([[2.0, -9.81, 0.0]], device=device)  # X方向2m/s²加速度，Y方向抵消重力
ang_accel = torch.zeros((1, 3), device=device)
```

### 定点悬停
```python
# 悬停在指定点 - 此处为原点上方1米
target_pos = torch.tensor([[0.0, 1.0, 0.0]], device=device)
current_pos = get_current_position()
current_vel = get_current_velocity()

# 使用PD控制器计算加速度
accel = (
    3.0 * (target_pos - current_pos) - 
    2.0 * current_vel + 
    torch.tensor([[0.0, -9.81, 0.0]], device=device)  # 抵消重力
)
```

## 参数调整

控制器的关键参数包括：

- `K_pos`：位置控制增益
- `K_vel`：速度控制增益
- `K_rot`：姿态控制增益
- `K_angvel`：角速度控制增益

这些参数可以根据无人机的物理特性和所需的控制性能进行调整。 