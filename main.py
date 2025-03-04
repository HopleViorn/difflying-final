import warp as wp

from flyinglib import Drone

def main():
    # single_drone_training_demo()
    drone = Drone('test')
    print(drone.props.shape)


if __name__ == "__main__":
    main()