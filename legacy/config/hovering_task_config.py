import torch
from flyinglib import FLYINGLIB_DIRECTORY


class task_config:
    """悬停任务的配置"""
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

    # 悬停参数
    hover_height = 0.5  # 悬停高度（米）
    hover_position_variance = 0.1  # 悬停位置随机方差

    # 奖励参数
    reward_parameters = {
        "position_reward_coefficient": 0.6,  # 位置奖励系数
        "attitude_reward_coefficient": 0.2,  # 姿态奖励系数
        "stability_reward_coefficient": 0.2,  # 稳定性奖励系数
        "dist_improvement_coefficient": 0.3,  # 距离改善奖励系数
        "hover_threshold": 0.1,  # 悬停阈值（米）
        "stable_threshold": 0.2,  # 稳定性阈值
        "success_threshold": 10,  # 成功阈值（连续步数）
    }

    # 课程学习配置
    class curriculum:
        min_level = 5
        max_level = 50
        check_after_log_instances = 1024
        increase_step = 5
        decrease_step = 2
        success_rate_for_increase = 0.75
        success_rate_for_decrease = 0.6

        def update_curriculim_level(self, success_rate, current_level):
            if success_rate > self.success_rate_for_increase:
                return min(current_level + self.increase_step, self.max_level)
            elif success_rate < self.success_rate_for_decrease:
                return max(current_level - self.decrease_step, self.min_level)
            return current_level 