import torch
from flyinglib import FLYINGLIB_DIRECTORY


class task_config:
    """前进飞行任务的配置"""
    seed = -1
    sim_name = "quad_sim"
    env_name = "empty_env"  # 使用空环境，无障碍物
    robot_name = "quadrotor"
    controller_name = "velocity_control"
    args = {}
    num_envs = 1024
    sim_steps = 300
    sim_dt = 0.02
    use_warp = True
    headless = True
    device = "cuda:0"
    observation_space_dim = 17  # position(3) + velocity(3) + orientation(4) + vector_to_target(3) + distance_to_target(1) + quad_angular_velocities(3)
    privileged_observation_space_dim = 0
    action_space_dim = 4  # x, y, z velocities + yaw rate
    high_level_control = True  # 使用高级控制模式

    return_state_before_reset = False  # 通常在重置后为下一个回合返回状态

    # 飞行参数
    flight_direction = [1.0, 0.0, 0.0]  # 飞行方向，默认沿x轴正方向
    flight_height = 0.5  # 飞行高度(米)
    flight_distance = 1.0  # 飞行距离目标(米)
    
    # 奖励参数
    reward_parameters = {
        "forward_reward_coefficient": 5.0,     # 前进奖励系数
        "height_reward_coefficient": 0.2,      # 高度控制奖励系数
        "lateral_reward_coefficient": 0.2,     # 轨迹控制奖励系数
        "stability_reward_coefficient": 0.1,   # 稳定性奖励系数
        "alignment_reward_coefficient": 0.2,   # 方向一致性奖励系数
        "completion_reward": 1.0,              # 完成任务奖励
        "height_error_scale": 5.0,             # 高度误差缩放因子
        "lateral_error_scale": 5.0,            # 横向误差缩放因子
        "hover_threshold": 0.1,                # 稳定阈值(米)
        "success_threshold": 5,                # 成功阈值(连续步数)
    }

    # 课程学习配置
    class curriculum:
        min_level = 10
        max_level = 50
        check_after_log_instances = 1024
        increase_step = 5
        decrease_step = 2
        success_rate_for_increase = 0.7
        success_rate_for_decrease = 0.5

        def update_curriculim_level(self, success_rate, current_level):
            if success_rate > self.success_rate_for_increase:
                return min(current_level + self.increase_step, self.max_level)
            elif success_rate < self.success_rate_for_decrease:
                return max(current_level - self.decrease_step, self.min_level)
            return current_level 