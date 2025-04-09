from flyinglib.rl_training.rl_games.rl_games_inference import MLP
from flyinglib.objects.drone import Drone
from flyinglib.simulation.simulator import SimpleConfig
from flyinglib.simulation.step import diff_step

import torch


def transform_policy_actions_to_high_level(actions):
    """
    将策略输出的动作转换为高级控制输入
    
    Args:
        actions: 形状为 (num_envs, 4) 的动作张量，范围在 [-1, 1] 之间
        
    Returns:
        形状为 (num_envs, 4) 的高级控制输入 [vx, vy, vz, yaw_rate]
    """
    # 定义各个维度的最大值
    max_velocity = torch.tensor([1.0, 1.0, 1.0], device="cuda:0")  # 最大速度 [vx, vy, vz] (m/s)
    max_yaw_rate = torch.tensor(1.0, device="cuda:0")  # 最大偏航角速率 (rad/s)
    
    # 将动作从 [-1, 1] 映射到 [vx, vy, vz, yaw_rate]
    high_level_actions = torch.zeros_like(actions)
    high_level_actions[:, :3] = actions[:, :3] * max_velocity.unsqueeze(0)
    high_level_actions[:, 3] = actions[:, 3] * max_yaw_rate
    return high_level_actions

def transform_policy_action(actions):
    """
    将策略输出的动作转换为推力控制输入
    
        Args:
            actions: 形状为 (num_envs, 4) 的动作张量，范围在 [-1, 1] 之间
            
        Returns:
            形状为 (num_envs, 4) 的推力控制输入，范围在 [0, 1] 之间
        """
        # 将动作从 [-1, 1] 映射到 [0, 1]
    transformed_actions = (actions + 1.0) * 0.5
    
    # 确保动作在 [0, 1] 范围内
    transformed_actions = torch.clamp(transformed_actions, 0.0, 1.0)
    
    return transformed_actions
        
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

    target_pos = [0.95, 0.95, 0.95]
    target_pos_tensor = torch.tensor([target_pos]).to("cuda:0")

    observation_space_dim = 17  # position(3) + velocity(3) + orientation(4) + vector_to_target(3) + distance_to_target(1) + quad_angular_velocities(3)
    action_space_dim = 4  #force

    policy = MLP(
        observation_space_dim,
        action_space_dim,
        policy_path
    ).to("cuda:0").eval()

    controller_config = SimpleConfig()
    controller_config.max_yaw_rate = 3.0  # 最大偏航角速率（弧度/秒）
    
    # 初始化高级控制器

    for _ in range(sim_steps):
        pos = q[:, :3] # positions
        att = q[:, 3:] # attitudes

        dp = target_pos_tensor - pos
        dist = torch.norm(dp, dim=1, keepdim=True)
        # 构建观察向量
        obs = torch.cat([
            pos,  # 位置 (3)
            qd[:, :3],  # 速度 (3)
            att,  # 姿态 (4)
            dp,  # 目标方向向量 (3)
            dist,  # 到目标的距离 (1)
            qd[:, 3:]  # 四旋翼角速度 (3)
        ], dim=1).to("cuda:0")
        
        # 使用策略网络生成动作
        a = policy(obs)
        a = transform_policy_action(a)

        print("action: ", a)

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

    test_render("/home/shuyi/difflying-final/flyinglib/rl_training/rl_games/runs/flying_navigation_ppo_08-17-17-14/nn/last_flying_navigation_ppo_ep_200_rew_-inf.pth", None)