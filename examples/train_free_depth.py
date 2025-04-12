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
from flyinglib.modules.policy import *
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

img_size = (64, 64)

class CNNImageEncoder(nn.Module):
    def __init__(self, image_res=(128, 128), latent_dims=64):
        super(CNNImageEncoder, self).__init__()
        self.image_res = image_res
        self.latent_dims = latent_dims
        
        # Feature extraction with stride convolutions
        self.features = nn.Sequential(
            # Block 1: [1, 135, 240] -> [32, 68, 120]
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            
            # Block 2: [32, 68, 120] -> [64, 34, 60]
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            
            # Block 3: [64, 34, 60] -> [128, 17, 30]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            
            # Block 4: [128, 17, 30] -> [256, 9, 15]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )
        
        # Final projection to latent dims
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [256, 1, 1]
            # nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(256, latent_dims, kernel_size=1),  # [64, 1, 1]
            nn.Flatten()  # [64]
        )

        # 添加sigmoid层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape input if needed
        if len(x.shape) == 3:  # [batch, H, W]
            x = x.unsqueeze(1)  # [batch, 1, H, W]
        
        # Forward pass
        features = self.features(x)
        latent = self.projection(features)
        return latent

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        
        self.image_encoder = CNNImageEncoder(image_res=img_size, latent_dims=64)
        
        # Original network with expanded input dimension (21 + 64 = 85)
        self.network = nn.Sequential(
            nn.Linear(input_dim + 64, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x, depth_imgs=None):
        if depth_imgs is not None:
            # Encode depth images using VAE
            depth_latent = self.image_encoder(depth_imgs)
            x = torch.cat([x, depth_latent], dim=1)
        else:
            x = torch.cat([x, torch.zeros(x.shape[0], 64, device=x.device)], dim=1)
        return self.network(x)

class GRUPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(GRUPolicyNetwork, self).__init__()
        
        self.image_encoder = CNNImageEncoder(image_res=img_size, latent_dims=64)
        self.input_dim = input_dim + 64
        self.hidden_size = hidden_size

        # GRU for temporal feature extraction
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=hidden_size, batch_first=True)

        # Fully connected head to output action
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ELU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, obs, depth_imgs, hidden_state=None):
        # depth_imgs: [B, 1, H, W]
        # obs: [B, D]

        depth_feat = self.image_encoder(depth_imgs)  # [B, 64]
        x = torch.cat([obs, depth_feat], dim=-1).unsqueeze(1)  # [B, 1, D+64]

        # Pass through GRU
        output, next_hidden = self.gru(x, hidden_state)  # output: [B, 1, H]
        action = self.fc(output.squeeze(1))  # [B, output_dim]
        return action, next_hidden


def train_free_flight(
    epochs: int = 100,
    batch_size: int = 128,
    sim_steps: int = 200,
    sim_dt: float = 0.04,
    initial_lr: float = 1e-3,
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
    drone = Drone('test', batch_size=batch_size, sim_steps=sim_steps, sim_dt=sim_dt)

    policy = GRUPolicyNetwork(input_dim=14, output_dim=6, hidden_size=128)

    # load_policy = "logs/test/20250412-022029/policy_final.pth"
    load_policy = None

    if load_policy:
        policy.load_state_dict(torch.load(load_policy))
        print(f"加载策略：{load_policy}")


    optimizer = optim.AdamW(policy.parameters(), lr=initial_lr, weight_decay=0.01)

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
        env_manager.setup_room(room_size=5.0, num_objects=16)
        env_manager.add_camera('front', config=camera_config, positions=q[:, :3], orientations=q[:, 3:])
        env_manager.set_camera_pose_tensor('front', q[:, :3], q[:, 3:])
        env_manager.capture_depth('front')

        # random_directions = torch.randn((batch_size, 3), device=DEVICE)
        # random_directions = random_directions / torch.norm(random_directions, dim=1, keepdim=True)  # 单位向量
        # random_distances = torch.rand((batch_size, 1), device=DEVICE) * 1.2
        # target_pos = random_directions * random_distances  # 最终目标位置

        x = torch.rand((batch_size, 1), device=DEVICE) * 0.1 + 0.2
        y = torch.rand((batch_size, 1), device=DEVICE) * 0.2 + 0.4
        z = torch.rand((batch_size, 1), device=DEVICE) * 0.5 + 1.3

        target_pos = torch.cat([x, y, z], dim=1)
        # print(f"target_pos: {target_pos}")

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

        hidden_state = None

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
                vel, # 3
                angular_vel, # 3
                att, # 4
                dir, # 3
                dist, # 1
                # pos, # 3
                # nearest_vec, # 3
                # nearest_dist, # 1
            ], dim=1).to(DEVICE)

            a, hidden_state = policy(obs, depth_imgs, hidden_state)
            a = action_transformation_function(a)
            a = controller.accelerations_to_motor_thrusts(a[:, :3], a[:, 3:], att)
            
            q, qd = diff_step(q, qd, a, drone)

            pos = q[:, :3] # positions
            att = q[:, 3:] # attitudes
            vel = qd[:, 3:] # velovoties
            rat = qd[:, :3] # rates

            dp = target_pos - pos
            dist = torch.norm(dp, dim=1, keepdim=True)

            # Compute loss
            loss_att += 1 - torch.mean(att[:, -1])
            loss_vel += torch.mean(vel**2)
            loss_vel_down += torch.mean(torch.relu(-vel - 1.0) ** 2)
            loss_extreme_angle += torch.mean(torch.relu(-att[:, 3] + 0.94))
            # loss_nearest_dist += torch.mean(torch.relu(nearest_dist - 0.1) ** 2)
            loss_nearest_dist += torch.mean(torch.relu(-nearest_dist + 0.4) ** 2) * 1

            # loss_pos += torch.mean(1 - torch.exp(-0.05*dist))
            loss_pos += torch.mean(dist)

        if epoch % 100 == 0:
            policy_path = log_path + f"policy_{epoch}_{dist.mean().item():.4f}.pth"
            torch.save(policy.state_dict(), policy_path)
        # Compute loss
        loss = 1 * loss_pos + 50 * loss_nearest_dist + 0.0000 * loss_att +  loss_vel * 0.00005 + 30 * torch.mean(dist) + loss_vel_down * 20 + loss_extreme_angle * 20

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

    policy_path = log_path + "policy_final.pth"
    torch.save(policy.state_dict(), policy_path)

    return policy_path


if __name__ == "__main__":
    policy_path = train_free_flight()
    print(f"Checkpoint saved at {policy_path}")
    print("开始测试策略...")
    final_pos = test_free(policy_path)
    print(f"测试完成！最终位置: {final_pos}")