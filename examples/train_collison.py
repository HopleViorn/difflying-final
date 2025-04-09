import numpy as np
from flyinglib.control.quad_lee_controller import QuadLeeController
import warp as wp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import datetime

from flyinglib.objects.propeller import *
from flyinglib.objects.drone import *
from flyinglib.simulation.step import *
from flyinglib.modules.policy import *



DEVICE = "cuda:0"
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def action_transformation_function(action):
        #限制动作范围
        # processed_action = (torch.sigmoid(action) - 0.5)
        processed_action = torch.clamp(action, -2 , 2)
        processed_action[:, 0:3] = processed_action[:, 0:3] * 0.5
        processed_action[:, 3:] = processed_action[:, 3:] * 0.0001
        # print(f"processed_action: {processed_action}")

        # # 限制加速度范围
        # processed_action[:, 0] = torch.clamp(processed_action[:, 0], -0.1, 0.1)
        # processed_action[:, 1] = torch.clamp(processed_action[:, 1], -0.1, 0.1)
        # processed_action[:, 2] = torch.clamp(processed_action[:, 2], -0.1, 0.1)
        # # 限制角加速度范围
        # processed_action[:, 3:] = torch.clamp(processed_action[:, 3:], -0.01, 0.01)

        return processed_action

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

def train_free_flight(
    epochs: int = 500,
    batch_size: int = 2048,
    sim_steps: int = 150,
    sim_dt: float = 0.02,
):  
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = "logs/test/" + date + "/"
    writer = SummaryWriter(log_path)

    drone = Drone('test', batch_size=batch_size, sim_steps=sim_steps, sim_dt=sim_dt)

    policy = PolicyNetwork(input_dim=17, output_dim=6)
    optimizer = optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0.0001)

    last_loss_value = np.inf

    # Training loop
    t = trange(epochs, desc='Train', leave=True)

    obstacles = [[0.5,0.5,0.5],
                 [-0.5,0.5,0.5],
                 [0.5,-0.5,0.5],
                 [-0.5,-0.5,0.5],
                 [0.5,0.5,-0.5],
                 [-0.5,0.5,-0.5],
                 [0.5,-0.5,-0.5],
                 [-0.5,-0.5,-0.5]]
    obstacles = np.array(obstacles)

    controller = QuadLeeController(num_envs=batch_size, device=DEVICE, drone=drone)

    for epoch in t:
        # Forward pass
        drone.step = 0

        init_pos = np.zeros((batch_size, 3), dtype=np.float32)
        init_att = np.tile(np.array([0., 0., 0., 1.], dtype=np.float32), (batch_size, 1))
        init_q = np.hstack((init_pos, init_att))
        init_qd = np.zeros((batch_size, 6), dtype=np.float32)

        q = torch.tensor(init_q, requires_grad=True)
        qd = torch.tensor(init_qd, requires_grad=True)        

        target_pos = torch.randn((batch_size, 3))

        # a = torch.zeros(drone.props.shape, requires_grad=True)
        
        loss_pos_total = 0
        loss_att_total = 0


        for _ in range(sim_steps):
            pos = q[:, :3] # positions
            att = q[:, 3:] # attitudes
            angular_vel = qd[:, :3] # angular velocities
            vel = qd[:, 3:] # velocities
            dp = target_pos - pos
            dist = torch.norm(dp, dim=1, keepdim=True)
            dir = dp / dist
            obs = torch.cat([
                pos, # 3
                vel, # 3
                angular_vel, # 3
                att, # 4
                dir, # 3
                dist, # 1
            ], dim=1).to(DEVICE)
            
            a = policy(obs)
            a = action_transformation_function(a)
            a = controller.accelerations_to_motor_thrusts(a[:, :3], a[:, 3:], att)
            
            q, qd = diff_step(q, qd, a, drone)
            dist = torch.norm(target_pos - q[:, :3], dim=1, keepdim=True)

            pos_loss = 0.3 * torch.exp(-dist ** 2)
            att_loss = 0.1 * torch.exp(-att[:, 3] ** 2)

            loss = pos_loss + att_loss
            loss = loss.mean()

            loss_pos_total += pos_loss.mean().detach().cpu().numpy().item()
            loss_att_total += att_loss.mean().detach().cpu().numpy().item()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        torch.cuda.empty_cache()

        loss_value = loss_pos_total + loss_att_total
        final_dist = torch.mean(dist).detach().cpu().numpy().item()

        writer.add_scalar("Loss/total", loss_value/sim_steps, epoch)
        writer.add_scalar("Loss/dist", final_dist, epoch)
        writer.add_scalar("Loss/pos", loss_pos_total/sim_steps, epoch)
        writer.add_scalar("Loss/att", loss_att_total/sim_steps, epoch)

        t.set_description(f"Loss: {loss_value}")
        

    writer.close()

    policy_path = log_path + "policy.pth"
    torch.save(policy.state_dict(), policy_path)

    # test_render(policy_path, obstacles)

    return policy_path


def test_render(
    policy_path: str,
    obstacles,
    sim_steps: int = 150,
    sim_dt: float = 0.02,
):
    print("evaluating...")
    drone = Drone('train', sim_steps=sim_steps, sim_dt=sim_dt, requires_grad=False)

    q = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]])
    qd = torch.zeros((1, 6))

    target_pos = [0.7, 0.7, 0.7]
    target_pos_tensor = torch.tensor([target_pos])

    policy = Towards()
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()

    for _ in range(sim_steps):
        pos = q[:, :3] # positions
        att = q[:, 3:] # attitudes

        dp = target_pos_tensor - pos
        dist = torch.norm(dp, dim=1, keepdim=True)
        dir = dp / dist

        a = policy(dir, dist, att, qd)
        q, qd = diff_step(q, qd, a, drone)
        drone.render(target_pos, obstacles)

    print(f"Final pose: {q.detach().cpu().numpy()}")
    print(f"Final speed: {qd.detach().cpu().numpy()}")

    drone.renderer.save()


if __name__ == "__main__":
    policy_path = train_free_flight()
    print(f"Checkpoint saved at {policy_path}")