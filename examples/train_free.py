import numpy as np
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


def train_free_flight(
    epochs: int = 500,
    batch_size: int = 512,
    sim_steps: int = 150,
    sim_dt: float = 0.02,
):  
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = "logs/test/" + date + "/"
    writer = SummaryWriter(log_path)

    drone = Drone('test', batch_size=batch_size, sim_steps=sim_steps, sim_dt=sim_dt)

    policy = Towards()
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    last_loss_value = np.infty

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

        target_pos = torch.randn((batch_size, 3))

        # a = torch.zeros(drone.props.shape, requires_grad=True)
        
        loss_pos = 0
        loss_att = 0
        loss_vel = 0

        pos = q[:, :3] # positions
        att = q[:, 3:] # attitudes

        dp = target_pos - pos
        dist = torch.norm(dp, dim=1, keepdim=True)
        dir = dp / dist

        for _ in range(sim_steps):
            a = policy(dir, dist, att, qd)
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
        
        
        loss_pos += torch.mean(dist)

        # Compute loss
        loss = 5. * loss_pos + 0.1 * loss_att + 0.001 * loss_vel

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        loss_value = loss.detach().cpu().numpy().item()
        loss_pos_value = loss_pos.detach().cpu().numpy().item()
        loss_att_value = loss_att.detach().cpu().numpy().item()
        loss_vel_value = loss_vel.detach().cpu().numpy().item()
        final_dist = torch.mean(dist).detach().cpu().numpy().item()

        writer.add_scalar("Loss/total", loss_value, epoch)
        writer.add_scalar("Loss/pos", loss_pos_value, epoch)
        writer.add_scalar("Loss/att", loss_att_value, epoch)
        writer.add_scalar("Loss/vel", loss_vel_value, epoch)
        writer.add_scalar("Loss/dist", final_dist, epoch)

        t.set_description(f"Loss: {loss_value}")
        
        if final_dist < 2e-2:
            break

    writer.close()

    policy_path = log_path + "policy.pth"
    torch.save(policy.state_dict(), policy_path)

    return policy_path


if __name__ == "__main__":
    policy_path = train_free_flight()
    print(f"Checkpoint saved at {policy_path}")