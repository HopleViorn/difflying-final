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
import argparse

from flyinglib.objects.propeller import *
from flyinglib.objects.drone import *
from flyinglib.simulation.step import *
from flyinglib.modules.policy import *
from flyinglib.scene.scene_manager import SceneManager

DEVICE = "cuda:0"
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from policy_nn import GRUPolicyNetwork
from policy_nn import img_size

def action_transformation_function(action):
        #限制动作范围
        processed_action = action
        processed_action[:, 0:3] = processed_action[:, 0:3] * 0.5
        processed_action[:, 3:] = processed_action[:, 3:] * 0.0001
        return processed_action

def train_free_flight(
    epochs: int = 100,
    batch_size: int = 128,
    sim_steps: int = 150,
    sim_dt: float = 0.06,
    initial_lr: float = 1e-3,
    load_policy_path: str = None,
):  
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = "logs/test/" + date + "/"
    writer = SummaryWriter(log_path)

    # 初始化环境管理器并设置场景
    
    # 添加前置摄像头
  
    camera_config = {
        'width': img_size[0],
        'height': img_size[1],
        'horizontal_fov_deg': 120,
        'max_range': 20.0,
        'calculate_depth': True,
        'num_sensors': batch_size,
        'segmentation_camera': False,
        'return_pointcloud': False
    }
    drone = Drone('test', batch_size=batch_size, sim_steps=sim_steps*3, sim_dt=sim_dt)

    policy = GRUPolicyNetwork(input_dim=10, output_dim=6, hidden_size=128)

    load_policy = load_policy_path

    if load_policy:
        policy.load_state_dict(torch.load(load_policy))
        print(f"加载策略：{load_policy}")

    optimizer = optim.AdamW(policy.parameters(), lr=initial_lr, weight_decay=0.0001)

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
        env_manager.setup_room()
        env_manager.add_camera('front', config=camera_config, positions=q[:, :3], orientations=q[:, 3:])
        env_manager.set_camera_pose_tensor('front', q[:, :3], q[:, 3:])
        env_manager.capture_depth('front')
        env_manager.generate_target_pos()

        
        loss_att = 0
        loss_vel = 0
        loss_vel_extreme = 0
        loss_extreme_angle = 0
        loss_nearest_dist = 0
        pos = q[:, :3] # positions
        att = q[:, 3:] # attitudes

        hidden_state = None

        for _ in range(sim_steps):
            # 更新所有环境的摄像机位置和朝向
            pos = q[:, :3] # positions
            att = q[:, 3:] # attitudes
            angular_vel = qd[:, :3] # angular velocities
            vel = qd[:, 3:] # velocities

            env_manager.set_camera_pose_tensor(
                'front',
                pos,
                att
            )

            depth_imgs = env_manager.capture_depth('front')
            depth_imgs = depth_imgs.squeeze(0)

            nearest_vec, nearest_dist = env_manager.get_nearest_object_distance(pos)

            obs = torch.cat([
                vel, # 3
                angular_vel, # 3
                att, # 4
            ], dim=1).to(DEVICE)

            a, hidden_state = policy(obs, depth_imgs, hidden_state)
            a = action_transformation_function(a)
            a = controller.accelerations_to_motor_thrusts(a[:, :3], a[:, 3:], att)
            
            q, qd = diff_step(q, qd, a, drone)
            q, qd = diff_step(q, qd, a, drone)
            q, qd = diff_step(q, qd, a, drone)

            pos = q[:, :3] # positions
            att = q[:, 3:] # attitudes
            vel = qd[:, 3:] # velovoties
            rat = qd[:, :3] # rates

            position = q[:, :3]

            # Compute loss
            target_velocity = torch.tensor([0, 0, 0.5], device=DEVICE)

            loss_att += 1 - torch.mean(att[:, -1])
            loss_vel += torch.mean(torch.relu(torch.norm(vel - target_velocity, dim=1) - 0.1))
            loss_vel_extreme += torch.mean(torch.relu(-vel - 1.0))
            loss_extreme_angle += torch.mean(torch.relu(-att[:, 3] + 0.94))
            loss_nearest_dist += torch.mean(torch.relu(-nearest_dist + 0.5) ** 2)

            # loss_pos += torch.mean(1 - torch.exp(-0.05*dist))

        if epoch % 100 == 0:
            policy_path = log_path + f"policy_{epoch}.pth"
            torch.save(policy.state_dict(), policy_path)

        loss = 0.1 * loss_att + 1 * loss_vel + 20 * loss_vel_extreme + 10 * loss_extreme_angle + 10 * loss_nearest_dist

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        loss_value = loss.detach().cpu().numpy().item()
        loss_att_value = loss_att.detach().cpu().numpy().item()
        loss_vel_value = loss_vel.detach().cpu().numpy().item()
        loss_vel_extreme_value = loss_vel_extreme.detach().cpu().numpy().item()
        loss_extreme_angle_value = loss_extreme_angle.detach().cpu().numpy().item()
        loss_nearest_dist_value = loss_nearest_dist.detach().cpu().numpy().item()

        writer.add_scalar("Loss/total", loss_value, epoch)
        writer.add_scalar("Loss/att", loss_att_value, epoch)
        writer.add_scalar("Loss/vel", loss_vel_value, epoch)
        writer.add_scalar("Loss/vel_extreme", loss_vel_extreme_value, epoch)
        writer.add_scalar("Loss/extreme_angle", loss_extreme_angle_value, epoch)
        writer.add_scalar("Loss/nearest_dist", loss_nearest_dist_value, epoch)
        
        # 更新学习率调度器
        scheduler.step(loss_value)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Training/learning_rate", current_lr, epoch)

        t.set_description(f"Loss: {loss_value:.4f}, att: {loss_att_value:.4f}, vel: {loss_vel_value:.4f}, vel_extreme: {loss_vel_extreme_value:.4f}, ex_angle: {loss_extreme_angle_value:.4f}, obs_dis: {loss_nearest_dist_value:.4f}, LR: {current_lr:.6f}")

    writer.close()

    policy_path = log_path + "policy_final.pth"
    torch.save(policy.state_dict(), policy_path)

    return policy_path

def test_free(policy_path: str, sim_steps: int = 150, sim_dt: float = 0.04):
    """Test trained policy and return final position"""
    # Initialize drone with batch_size=1
    drone = Drone('test', batch_size=1, sim_steps=sim_steps, sim_dt=sim_dt)
    
    # Initialize policy
    policy = GRUPolicyNetwork(input_dim=10, output_dim=6, hidden_size=128)
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()
    
    # Initialize controller
    controller = QuadLeeController(num_envs=1, device=DEVICE, drone=drone)
    
    # Initialize scene manager
    env_manager = SceneManager(batch_size=1)
    env_manager.setup_room(room_size=6.0, num_objects=32)
    
    # Add camera matching training config
    camera_config = {
        'width': img_size[0],
        'height': img_size[1],
        'horizontal_fov_deg': 120,
        'max_range': 20.0,
        'calculate_depth': True,
        'num_sensors': 1,
        'segmentation_camera': False,
        'return_pointcloud': False
    }
    
    # Initialize state
    q = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], device=DEVICE)
    qd = torch.zeros((1, 6), device=DEVICE)
    hidden_state = None
    env_manager.add_camera('front', config=camera_config, positions=q[:, :3], orientations=q[:, 3:])
    
    # Run simulation
    for _ in range(sim_steps):
        pos = q[:, :3]
        att = q[:, 3:]
        angular_vel = qd[:, :3]
        vel = qd[:, 3:]
        
        # Update camera pose
        env_manager.set_camera_pose_tensor('front', pos, att)
        depth_imgs = env_manager.capture_depth('front')
        depth_imgs = depth_imgs.squeeze(0)
        
        # Build observation (matches training)
        obs = torch.cat([
            vel,  # 3
            angular_vel,  # 3
            att,  # 4
        ], dim=1).to(DEVICE)
        
        # Get action from policy
        with torch.no_grad():
            a, hidden_state = policy(obs, depth_imgs, hidden_state)
            a = action_transformation_function(a)
            a = controller.accelerations_to_motor_thrusts(a[:, :3], a[:, 3:], att)
        
        # Simulation step
        q, qd = diff_step(q, qd, a, drone)
    
    return q[:, :3].detach().cpu().numpy()[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练和测试无人机飞行策略')
    parser.add_argument('--path', type=str, help='预训练权重文件路径', default=None)
    parser.add_argument('--epochs', type=int, help='训练轮数', default=100)
    args = parser.parse_args()
    
    policy_path = train_free_flight(load_policy_path=args.path, epochs=args.epochs)
    print(f"Checkpoint saved at {policy_path}")
    print("开始测试策略...")
    final_pos = test_free(policy_path)
    print(f"测试完成！最终位置: {final_pos}")