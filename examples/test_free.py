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


def test_render(
    policy_path: str,
    sim_steps: int = 150,
    sim_dt: float = 0.02,
):
    drone = Drone('test', sim_steps=sim_steps, sim_dt=sim_dt, requires_grad=False)

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
        drone.render(target_pos)

    print(f"Final pose: {q.detach().cpu().numpy()}")
    print(f"Final speed: {qd.detach().cpu().numpy()}")

    drone.renderer.save()



if __name__ == "__main__":
    test_render(policy_path="logs/test/20241105-154547/policy.pth")
