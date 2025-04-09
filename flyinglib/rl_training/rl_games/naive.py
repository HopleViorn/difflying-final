from flyinglib.control.quad_lee_controller import QuadLeeController
from flyinglib.rl_training.rl_games.rl_games_inference import MLP
from flyinglib.objects.drone import Drone
from flyinglib.simulation.config.navigation_task_config import task_config
from flyinglib.simulation.step import diff_step
from flyinglib.simulation import task_registry

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from datetime import datetime

from flyinglib.simulation.tasks.navigation_task import NavigationTask

# 简单的策略网络定义
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def train(
    sim_steps: int = 150,
    sim_dt: float = 0.02,
    num_envs: int = 2048,
    num_epochs: int = 200  # 训练迭代次数
    lr = 0.0005  # 学习率
):
    # 配置参数
    device = task_config.device
    print(f"训练配置: 设备={device}, 环境数={num_envs}, 迭代次数={num_iterations}")

    drone = Drone('train', sim_steps=sim_steps, sim_dt=sim_dt, requires_grad=True)
    
    # 创建输出目录
    output_dir = os.path.join("trained_models", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"模型将保存到: {output_dir}")
    
    # 创建策略网络
    policy = PolicyNetwork(
        input_dim=17,
        output_dim=6
    ).to(device)
    
    # 创建优化器
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    controller = QuadLeeController(num_envs=num_envs, device=device, drone=drone)
    
    # 记录训练信息
    rewards_history = []

    q = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]]).to(device)
    qd = torch.zeros((1, 6)).to(device)
    
    policy.train()
    with torch.enable_grad():
        for epoch in range(num_epochs):
            
            for i in range(sim_steps):

                vec_to_target = torch.randn((num_envs, 3), device=device, requires_grad=True)
                dist_to_target = torch.norm(vec_to_target, dim=1, keepdim=True)
                dir = vec_to_target / dist_to_target

                obs = torch.cat([
                    q[:, :3],
                    qd[:, 3:],
                    q[:, 3:],
                    dir, dist_to_target, qd[:, :3]], dim=1).to(device)
                
                action = policy(obs)
                action = task_config.action_transformation_function(action)

                action = controller.accelerations_to_motor_thrusts(action[:, :3], action[:, 3:], att)
                
                next_obs, rewards, terminated, truncated, info = task.step(action)
                

                rewards = task.sim.positions
                
                # 更新策略网络
                optimizer.step()
                optimizer.zero_grad()
                
            
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, "policy_final.pth")
    torch.save(policy.state_dict(), final_model_path)
    print(f"训练完成! 最终模型已保存到 {final_model_path}")
    
    # 记录奖励历史
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Training Rewards')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(output_dir, "rewards.png"))
    
    return policy


def test_render(
    policy_path: str,
    obstacles,
    sim_steps: int = 150,
    sim_dt: float = 0.02,
):
    print("evaluating...")
    drone = Drone('train', sim_steps=sim_steps, sim_dt=sim_dt, requires_grad=False)

    q = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]]).to("cuda:0")
    qd = torch.zeros((1, 6)).to("cuda:0")

    target_pos = [0, 0.95, 0]
    target_pos_tensor = torch.tensor([target_pos]).to("cuda:0")

    observation_space_dim = 17  # position(3) + velocity(3) + orientation(4) + vector_to_target(3) + distance_to_target(1) + quad_angular_velocities(3)
    action_space_dim = 6  #force

    policy = MLP(
        observation_space_dim,
        action_space_dim,
        policy_path
    ).to("cuda:0").eval()

    controller = QuadLeeController(num_envs=1, device="cuda:0", drone=drone)

    for _ in range(sim_steps):
        pos = q[:, :3] # positions
        att = q[:, 3:] # attitudes

        dp = target_pos_tensor - pos
        dist = torch.norm(dp, dim=1, keepdim=True)
        dp = dp / dist
        # 构建观察向量
        obs = torch.cat([
            pos,  # 位置 (3)
            qd[:, 3:],  # 速度 (3)
            att,  # 姿态 (4)
            dp,  # 目标方向向量 (3)
            dist,  # 到目标的距离 (1)
            qd[:, :3]  # 四旋翼角速度 (3)
        ], dim=1).to("cuda:0")
        
        # 使用策略网络生成动作
        a = policy(obs)
        a = task_config.action_transformation_function(a)
        print("action: ", a)

        a = controller.accelerations_to_motor_thrusts(a[:, :3], a[:, 3:], att)

        position = q[:, :3]
        velocity = qd[:, 3:]
        angular_velocity = qd[:, :3]
        attitude = q[:, 3:]
        
        print("position: ", position.detach().cpu().numpy())
        print("velocity: ", velocity.detach().cpu().numpy())
        print("attitude: ", attitude.detach().cpu().numpy())
        print("angular_velocity: ", angular_velocity.detach().cpu().numpy())

        # a = torch.tensor([[0.32849591, 0.32849591, 0.32849591, 0.32849591]], device="cuda:0")
        print("action: ", a)        

        q, qd = diff_step(q, qd, a, drone)
        print("static action: ", drone.static_action)

        drone.render(target_pos, obstacles)

    print(f"Final position: {q[:, :3].detach().cpu().numpy()}")

    print(f"Final pose: {q.detach().cpu().numpy()}")
    print(f"Final speed: {qd.detach().cpu().numpy()}")

    drone.renderer.save()


if __name__ == "__main__":
    # 训练新模型
    policy = train()
    
    # 测试现有模型
    # test_render("runs/flying_navigation_ppo_09-19-10-02/nn/last_flying_navigation_ppo_ep_500_rew_-inf.pth", None)