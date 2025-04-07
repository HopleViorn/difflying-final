import numpy as np
import warp as wp
import warp.sim
import warp.sim.render

from flyinglib.objects.propeller import *


class Drone:
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        sim_steps: int = 100,
        sim_dt: float = 0.01,
        size: float = 0.2,
        requires_grad: bool = True,
        decay: float = 1.0,
    ) -> None:
        self.batch_size = batch_size
        self.sim_steps = sim_steps
        self.sim_dt = sim_dt
        self.fps = 1 / sim_dt
        self.requires_grad = requires_grad
        self.discount = np.exp(-decay * sim_dt)

        # Current tick of the simulation
        self.step = 0

        # Initialize the helper to build a physics scene.
        builder = wp.sim.ModelBuilder()
        builder.rigid_contact_margin = 0.05

        # Initialize the rigid bodies, propellers, and colliders.
        props = []
        # colliders = []
        crossbar_length = size
        crossbar_height = size * 0.05
        crossbar_width = size * 0.05
        carbon_fiber_density = 1750.0  # kg / m^3

        for i in range(batch_size):
            # Register the drone as a rigid body in the simulation model.
            body = builder.add_body(name=f"{name}_{i}")

            # Define the shapes making up the drone's rigid body.
            builder.add_shape_box(
                body,
                hx=crossbar_length,
                hy=crossbar_height,
                hz=crossbar_width,
                density=carbon_fiber_density,
                collision_group=i,
            )
            builder.add_shape_box(
                body,
                hx=crossbar_width,
                hy=crossbar_height,
                hz=crossbar_length,
                density=carbon_fiber_density,
                collision_group=i,
            )
            
            # Initialize the propellers.
            props.extend(
                (
                    define_propeller(
                        body,
                        wp.vec3(crossbar_length, 0.0, 0.0),
                        turning_direction=-1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(-crossbar_length, 0.0, 0.0),
                        turning_direction=1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(0.0, 0.0, crossbar_length),
                        turning_direction=1.0,
                    ),
                    define_propeller(
                        body,
                        wp.vec3(0.0, 0.0, -crossbar_length),
                        turning_direction=-1.0,
                    ),
                ),
            )

            # colliders.append(
            #     (
            #         builder.add_shape_capsule(
            #             -1,
            #             pos=(0.5, 2.0, 0.5),
            #             radius=0.15,
            #             half_height=2.0,
            #             collision_group=i,
            #         ),
            #     ),
            # )
            self.create_environment()

        # Build the model and set-up its properties.
        self.model = builder.finalize(requires_grad=requires_grad)
        self.model.ground = False
        self.num_props = int(len(props) / self.batch_size)
        self.props = wp.array(props, dtype=Propeller, shape=(self.batch_size, self.num_props))
        # self.colliders = wp.array(colliders, dtype=int)

        # Use the Euler integrator for stepping through the simulation.
        self.integrator = wp.sim.SemiImplicitIntegrator()

        # Floating
        self.mass = self.model.body_mass.numpy()[0]
        gravity = np.linalg.norm(self.model.gravity)
        self.max_thrust = props[0].max_thrust
        action = (self.mass * gravity) / (self.num_props * self.max_thrust)
        self.static_action = action * np.ones(self.num_props)

        # Initialize the required simulation states.
        self.states = tuple(self.model.state() for _ in range(self.sim_steps + 1))

        self.renderer = wp.sim.render.SimRenderer(self.model, f"drone_{name}.usd", fps=self.fps)

    @property
    def state(self) -> wp.sim.State:
        return self.states[self.step]

    @property
    def next_state(self) -> wp.sim.State:
        return self.states[self.step + 1]
    
    def create_environment(self):
        return

    def render(self, target_pos, obstacles=None):
        self.renderer.begin_frame(self.step * self.sim_dt)
        self.renderer.render(self.state)

        # Render a sphere as the current target.
        self.renderer.render_sphere(
            "target",
            target_pos,
            wp.quat_identity(),
            0.05,
            color=(1.0, 0.0, 0.0),
        )

        # Render the obstacles.
        if obstacles is not None:
            for i, obs in enumerate(obstacles):
                self.renderer.render_sphere(
                    f"obstacle_{i}",
                    obs,
                    wp.quat_identity(),
                    0.05,
                    color=(0.0, 0.0, 1.0),
                )

        self.renderer.end_frame()


if __name__ == "__main__":
    batch_size = 5

    drone = Drone('test', batch_size=batch_size)
    print(drone.static_action)