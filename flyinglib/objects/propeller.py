import warp as wp


@wp.struct
class Propeller:
    body: int
    pos: wp.vec3
    dir: wp.vec3
    thrust: float
    power: float
    diameter: float
    height: float
    max_rpm: float
    max_thrust: float
    max_torque: float
    turning_direction: float
    max_speed_square: float


def define_propeller(
    drone: int,
    pos: wp.vec3,
    thrust: float = 0.109919,
    power: float = 0.040164,
    diameter: float = 0.2286,
    height: float = 0.01,
    max_rpm: float = 6396.667,
    turning_direction: float = 1.0,
):
    air_density = 1.225
    rps = max_rpm / 60
    rps_square = rps**2

    prop = Propeller()
    prop.body = drone
    prop.pos = pos
    prop.dir = wp.vec3(0.0, 1.0, 0.0)
    prop.thrust = thrust
    prop.power = power
    prop.diameter = diameter
    prop.height = height
    prop.max_rpm = max_rpm
    prop.turning_direction = turning_direction
    # Corresponding to control signal 1
    prop.max_thrust = thrust * air_density * rps_square * diameter**4
    prop.max_torque = power * air_density * rps_square * diameter**5 / wp.TAU

    return prop


@wp.kernel
def compute_prop_wrenches(
    body_q: wp.array(dtype=wp.transform),
    action: wp.array2d(dtype=float),
    props: wp.array2d(dtype=Propeller),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    body_id, prop_id = wp.tid()
    tf = body_q[body_id]
    control = action[body_id][prop_id]
    prop = props[body_id][prop_id]
    
    dir = wp.transform_vector(tf, prop.dir)
    force = dir * prop.max_thrust * control
    torque = dir * prop.max_torque * control * prop.turning_direction
    moment_arm = wp.transform_vector(tf, prop.pos)
    torque += wp.cross(moment_arm, force)
    torque *= 0.8

    wp.atomic_add(body_f, body_id, wp.spatial_vector(torque, force))
