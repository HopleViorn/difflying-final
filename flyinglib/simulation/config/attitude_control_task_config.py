import torch
import math
from flyinglib import FLYINGLIB_DIRECTORY


class task_config:
    """姿态控制任务的配置"""
    seed = -1
    sim_name = "quad_sim"
    env_name = "empty_env"  # 使用空环境，无障碍物
    robot_name = "quadrotor"
    controller_name = "thrust_control"  # 使用低级推力控制
    args = {}
    num_envs = 1024
    sim_steps = 300
    sim_dt = 0.02
    use_warp = True
    headless = True
    device = "cuda:0"
    observation_space_dim = 17  # position(3) + velocity(3) + orientation(4) + vector_to_target(3) + distance_to_target(1) + quad_angular_velocities(3)
    privileged_observation_space_dim = 0
    action_space_dim = 4  # 4个推力输入
    high_level_control = False  # 使用低级控制模式

    return_state_before_reset = False  # 通常在重置后为下一个回合返回状态

    # 姿态控制参数
    hover_height = 0.5  # 悬停高度（米）
    attitude_change_prob = 0.005  # 目标姿态变化概率
    max_attitude_angle = 0.3  # 最大姿态角度（弧度，约17度）

    # 阶段设置
    phase_1_max_yaw = math.pi  # 阶段1最大偏航角（-π到π）
    phase_2_max_pitch = 0.3  # 阶段2最大俯仰角（弧度）
    phase_3_max_roll = 0.3  # 阶段3最大横滚角（弧度）

    # 奖励参数
    reward_parameters = {
        "position_reward_scale": -5.0,        # 位置奖励缩放因子
        "attitude_reward_max_error": math.pi,  # 姿态奖励最大误差
        "stability_reward_scale": -2.0,       # 稳定性奖励缩放因子
        "improvement_reward_scale": 2.0,      # 改善奖励缩放因子
        "phase_1_weights": [0.4, 0.3, 0.2, 0.1],  # 阶段1权重 [位置, 姿态, 稳定性, 改善]
        "phase_2_weights": [0.3, 0.4, 0.2, 0.1],  # 阶段2权重
        "phase_3_weights": [0.2, 0.5, 0.2, 0.1],  # 阶段3权重
        "phase_1_attitude_threshold": 0.2,    # 阶段1姿态误差阈值
        "phase_2_attitude_threshold": 0.15,   # 阶段2姿态误差阈值
        "phase_3_attitude_threshold": 0.1,    # 阶段3姿态误差阈值
        "position_threshold": 0.1,            # 位置误差阈值
        "stability_threshold": 0.2,           # 稳定性阈值
        "success_threshold": 20,              # 成功阈值（连续步数）
        "phase_step": 0.25,                   # 每次成功后的阶段进步幅度
    }

    # 课程学习配置
    class curriculum:
        min_level = 5
        max_level = 30
        check_after_log_instances = 1024
        increase_step = 3
        decrease_step = 1
        success_rate_for_increase = 0.65
        success_rate_for_decrease = 0.45

        def update_curriculim_level(self, success_rate, current_level):
            if success_rate > self.success_rate_for_increase:
                return min(current_level + self.increase_step, self.max_level)
            elif success_rate < self.success_rate_for_decrease:
                return max(current_level - self.decrease_step, self.min_level)
            return current_level 