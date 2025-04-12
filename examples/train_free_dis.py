import numpy as np
from flyinglib.control.quad_lee_controller import QuadLeeController
import warp as wp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
import datetime
import matplotlib.pyplot as plt

from flyinglib.objects.propeller import *
from flyinglib.objects.drone import *
from flyinglib.simulation.step import *
from flyinglib.scene.scene_manager import SceneManager


DEVICE = "cuda:0"
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def action_transformation_function(action):
        #限制动作范围
        # processed_action = (torch.sigmoid(action) - 0.5)
        processed_action = action
        processed_action[:, 0:3] = processed_action[:, 0:3] * 0.5
        processed_action[:, 3:] = processed_action[:, 3:] * 0.0001
        # processed_action[:, 0:3] = torch.clamp(processed_action[:, 0:3], -0.5, 0.5)
        # processed_action[:, 3:] = torch.clamp(processed_action[:, 3:], -0.0005, 0.0005)

        # print(f"processed_action: {processed_action}")

        # # 限制加速度范围
        # processed_action[:, 0] = torch.clamp(processed_action[:, 0], -0.1, 0.1)
        # processed_action[:, 1] = torch.clamp(processed_action[:, 1], -0.1, 0.1)
        # processed_action[:, 2] = torch.clamp(processed_action[:, 2], -0.1, 0.1)
        # # 限制角加速度范围
        # processed_action[:, 3:] = torch.clamp(processed_action[:, 3:], -0.01, 0.01)

        return processed_action

img_size = 64

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def train_free_flight(
    epochs: int = 1000,
    batch_size: int = 64,
    sim_steps: int = 200,
    sim_dt: float = 0.02,
    initial_lr: float = 1e-2,
):  
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = "logs/test/" + date + "/"
    writer = SummaryWriter(log_path)

    # 初始化环境管理器并设置场景
    
    # 添加前置摄像头
  
    camera_config = {
        'width': 64,
        'height': 64,
        'horizontal_fov_deg': 120,
        'max_range': 20.0,
        'calculate_depth': True,
        'num_sensors': batch_size,
        'segmentation_camera': False,
        'return_pointcloud': False
    }
    drone = Drone('test', batch_size=batch_size, sim_steps=sim_steps, sim_dt=sim_dt)

    policy = PolicyNetwork(input_dim=21, output_dim=6)
    optimizer = optim.AdamW(policy.parameters(), lr=initial_lr, weight_decay=0.0001)
    
    # 添加学习率调度器，在验证损失停止改善时降低学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,    
                                 verbose=True, min_lr=1e-4)

    controller = QuadLeeController(num_envs=batch_size, device=DEVICE, drone=drone)

    # Training loop
    t = trange(epochs, desc='Train', leave=True)  

    for epoch in t:
        # Forward pass
        drone.step = 0

        init_pos = np.zeros((batch_size, 3), dtype=np.float32)
        init_att = np.tile(np.array([0., 0., 0., 1.], dtype=np.float32), (batch_size, 1))
        init_q = np.hstack((init_pos, init_att))
        init_qd = np.zeros((batch_size, 6), dtype=np.float32)

        q = torch.tensor(init_q, requires_grad=True)
        qd = torch.tensor(init_qd, requires_grad=True)        

        env_manager = SceneManager(batch_size=batch_size)
        env_manager.setup_room(room_size=5.0, num_objects=32)
        env_manager.save_scene(log_path + "scene.json")
        env_manager.add_camera('front', config=camera_config, positions=q[:, :3], orientations=q[:, 3:])
        env_manager.set_camera_pose_tensor('front', q[:, :3], q[:, 3:])
        env_manager.capture_depth('front')


        random_directions = torch.randn((batch_size, 3), device=DEVICE)
        random_directions = random_directions / torch.norm(random_directions, dim=1, keepdim=True)  # 单位向量
        random_distances = torch.rand((batch_size, 1), device=DEVICE) * 1.0 + 1.0  # 2到3米之间的随机距离
        target_pos = random_directions * random_distances  # 最终目标位置


        # a = torch.zeros(drone.props.shape, requires_grad=True)
        
        loss_pos = 0
        loss_att = 0
        loss_vel = 0
        loss_vel_down = 0
        loss_extreme_angle = 0
        loss_nearest_dist = 0
        pos = q[:, :3] # positions
        att = q[:, 3:] # attitudes

        dp = target_pos - pos
        dist = torch.norm(dp, dim=1, keepdim=True)
        dir = dp / dist

        for _ in range(sim_steps):
            # 更新所有环境的摄像机位置和朝向
            
            pos = q[:, :3] # positions
            att = q[:, 3:] # attitudes
            angular_vel = qd[:, :3] # angular velocities
            vel = qd[:, 3:] # velocities
            dp = target_pos - pos
            dist = torch.norm(dp, dim=1, keepdim=True)
            dir = dp / dist

            env_manager.set_camera_pose_tensor(
                'front',
                pos,
                att
            )

            depth_imgs = env_manager.capture_depth('front')
            depth_imgs = depth_imgs.squeeze(0)
            #save
            # plt.imshow(depth_imgs[0].cpu().numpy())
            # plt.savefig(f"depth_imgs_{0}.png")
            # plt.close()
            # depth_imgs = depth_imgs.squeeze(0)  

            # print(f"nearest_vec: {nearest_vec}, nearest_dist: {nearest_dist}")
            nearest_vec, nearest_dist = env_manager.get_nearest_object_distance(pos)

            obs = torch.cat([
                pos, # 3
                vel, # 3
                angular_vel, # 3
                att, # 4
                dir, # 3
                dist, # 1
                nearest_vec, # 3
                nearest_dist, # 1
            ], dim=1).to(DEVICE)

            a = policy(obs)
            a = action_transformation_function(a)
            a = controller.accelerations_to_motor_thrusts(a[:, :3], a[:, 3:], att)
            
            q, qd = diff_step(q, qd, a, drone)

            pos = q[:, :3] # positions
            att = q[:, 3:] # attitudes
            vel = qd[:, 3:] # velovoties
            rat = qd[:, :3] # rates

            dp = target_pos - pos
            dist = torch.norm(dp, dim=1, keepdim=True)
            dir = dp / dist

            # Compute loss
            loss_att += 1 - torch.mean(att[:, -1])
            loss_vel += torch.mean(vel**2)
            loss_vel_down += torch.mean(torch.relu(-vel - 1.4) ** 2)
            loss_extreme_angle += torch.mean(torch.relu(-att[:, 3] + 0.95))
            # loss_nearest_dist += torch.mean(torch.relu(nearest_dist - 0.1) ** 2)
            loss_nearest_dist += torch.mean(torch.relu(-nearest_dist + 0.3) ** 2)

            # loss_pos += torch.mean(1 - torch.exp(-0.05*dist))
            loss_pos += torch.mean(dist)
        # Compute loss
        loss = 0.1 * loss_pos + 10 * loss_nearest_dist + 0.01 * loss_att +  loss_vel*0.001 + 5 * torch.mean(dist) + loss_vel_down * 20 + loss_extreme_angle * 20

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        loss_value = loss.detach().cpu().numpy().item()
        loss_pos_value = loss_pos.detach().cpu().numpy().item()
        loss_att_value = loss_att.detach().cpu().numpy().item()
        loss_vel_value = loss_vel.detach().cpu().numpy().item()
        loss_vel_down_value = loss_vel_down.detach().cpu().numpy().item()
        loss_extreme_angle_value = loss_extreme_angle.detach().cpu().numpy().item()
        loss_nearest_dist_value = loss_nearest_dist.detach().cpu().numpy().item()
        final_dist = torch.mean(dist).detach().cpu().numpy().item()

        writer.add_scalar("Loss/total", loss_value, epoch)
        writer.add_scalar("Loss/pos", loss_pos_value, epoch)
        writer.add_scalar("Loss/att", loss_att_value, epoch)
        writer.add_scalar("Loss/vel", loss_vel_value, epoch)
        writer.add_scalar("Loss/dist", final_dist, epoch)
        
        # 更新学习率调度器
        scheduler.step(loss_value)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Training/learning_rate", current_lr, epoch)

        t.set_description(f"Loss: {loss_value:.4f}, pos: {loss_pos_value:.4f}, att: {loss_att_value:.4f}, vel: {loss_vel_value:.4f}, vel_down: {loss_vel_down_value:.4f}, ex_angle: {loss_extreme_angle_value:.4f}, obs_dis: {loss_nearest_dist_value:.4f}, final dist: {final_dist:.4f}, LR: {current_lr:.6f}")

    writer.close()

    policy_path = log_path + "policy.pth"
    torch.save(policy.state_dict(), policy_path)

    return policy_path


def test_free(
    policy_path: str,
    sim_steps: int = 150,
    sim_dt: float = 0.02,
):
    """
    测试训练好的策略网络并渲染结果
    
    Args:
        policy_path: 策略网络权重的路径
        sim_steps: 模拟步数
        sim_dt: 模拟时间步长
    """
    # 初始化环境管理器并设置场景
    env_manager = SceneManager(batch_size=1)
    env_manager.setup_room(room_size=5.0, num_objects=4)
    
    # 创建单个无人机实例，不需要梯度计算
    drone = Drone('test', batch_size=1, sim_steps=sim_steps, sim_dt=sim_dt, requires_grad=False)
    
    # 初始化位置和姿态
    q = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]]).to(DEVICE)
    qd = torch.zeros((1, 6)).to(DEVICE)
    
    # 设置目标位置
    target_pos = [0.7, 0.7, 0.7]
    target_pos_tensor = torch.tensor([target_pos]).to(DEVICE)
    
    # 加载策略网络
    policy = PolicyNetwork(input_dim=21, output_dim=6)
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()
    
    # 创建控制器
    controller = QuadLeeController(num_envs=1, device=DEVICE, drone=drone)
    
    # 模拟并渲染每一步
    for _ in range(sim_steps):
        pos = q[:, :3]  # 位置
        att = q[:, 3:]  # 姿态
        angular_vel = qd[:, :3]  # 角速度
        vel = qd[:, 3:]  # 速度
        
        # 计算到目标的方向和距离
        dp = target_pos_tensor - pos
        dist = torch.norm(dp, dim=1, keepdim=True)
        dir = dp / dist
        
        # 获取最近物体距离
        nearest_vec, nearest_dist = env_manager.get_nearest_object_distance(pos)
        
        # 构建观察向量
        obs = torch.cat([
            pos,  # 位置 (3)
            vel,  # 速度 (3)
            angular_vel,  # 角速度 (3)
            att,  # 姿态 (4)
            dir,  # 方向 (3)
            dist,  # 距离 (1)
            nearest_vec,  # 最近物体方向 (3)
            nearest_dist,  # 最近物体距离 (1)
        ], dim=1).to(DEVICE)
        
        # 使用策略网络生成动作
        with torch.no_grad():
            a = policy(obs)
            a = action_transformation_function(a)
            a = controller.accelerations_to_motor_thrusts(a[:, :3], a[:, 3:], att)
        
        # 模拟一步
        q, qd = diff_step(q, qd, a, drone)
        
        # 渲染当前状态
        drone.render(target_pos)
    
    # 打印最终位置和速度
    print(f"最终位置: {q[:, :3].detach().cpu().numpy()}")
    print(f"最终姿态: {q.detach().cpu().numpy()}")
    print(f"最终速度: {qd.detach().cpu().numpy()}")
    
    # 保存渲染结果
    drone.renderer.save()
    
    return q[:, :3].detach().cpu().numpy()


if __name__ == "__main__":
    policy_path = train_free_flight()
    print(f"Checkpoint saved at {policy_path}")
    print("开始测试策略...")
    final_pos = test_free(policy_path)
    print(f"测试完成！最终位置: {final_pos}")