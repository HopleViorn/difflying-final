import numpy as np
import warp as wp
import torch

from flyinglib.objects.propeller import *
from flyinglib.objects.drone import *


class StepLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, body_q, body_qd, action, drone):
        ctx.tape = wp.Tape()
        ctx.body_q = wp.from_torch(body_q, dtype=wp.transform)
        ctx.body_qd = wp.from_torch(body_qd, dtype=wp.spatial_vector)

        ctx.action = wp.from_torch(action)
        ctx.discount = drone.discount

        with ctx.tape:
            drone.state.clear_forces()
            drone.state.body_q = ctx.body_q
            drone.state.body_qd = ctx.body_qd

            wp.launch(
                compute_prop_wrenches,
                dim=drone.props.shape,
                inputs=(
                    ctx.body_q,
                    ctx.action,
                    drone.props,
                    drone.model.body_com,
                ),
                outputs=(drone.state.body_f,),
            )
            
            drone.integrator.simulate(
                drone.model,
                drone.state,
                drone.next_state,
                drone.sim_dt,
            )

        ctx.next_body_q = drone.next_state.body_q
        ctx.next_body_qd = drone.next_state.body_qd
        
        drone.step += 1

        return (wp.to_torch(ctx.next_body_q), wp.to_torch(ctx.next_body_qd))

    @staticmethod
    def backward(ctx, adj_next_body_q, adj_next_body_qd):
        ctx.next_body_q.grad = wp.from_torch(adj_next_body_q, dtype=wp.transform)
        ctx.next_body_qd.grad = wp.from_torch(adj_next_body_qd, dtype=wp.spatial_vector)

        ctx.tape.backward()
        
        adj_body_q = wp.to_torch(ctx.body_q.grad).clone() * ctx.discount
        adj_body_qd = wp.to_torch(ctx.body_qd.grad).clone() * ctx.discount
        adj_action = wp.to_torch(ctx.action.grad).clone() * ctx.discount

        ctx.tape.zero()

        return (adj_body_q, adj_body_qd, adj_action, None)


def diff_step(q, qd, a, drone):
    return StepLayer.apply(q, qd, a, drone)


if __name__ == "__main__":
    DEVICE = "cuda:0"
    batch_size = 5
    sim_steps = 100

    drone = Drone('test', batch_size=batch_size, sim_steps=sim_steps)

    init_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    init_qd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    init_q = torch.tensor([init_q for _ in range(batch_size)], device=DEVICE, requires_grad=True)
    init_qd = torch.tensor([init_qd for _ in range(batch_size)], device=DEVICE, requires_grad=True)
    a = torch.zeros(drone.props.shape, device=DEVICE, requires_grad=True)

    q = init_q
    qd = init_qd

    print(q)
    print(qd)

    for _ in range(100):
        q, qd = StepLayer.apply(q, qd, a, drone)

    print(q)
    print(qd)

    loss = torch.mean(q, dim=0)[1]
    loss_value = loss.detach().cpu().numpy().item()
    
    print(loss_value)

    loss.backward()

    print(init_q.grad)
    print(init_qd.grad)