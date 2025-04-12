import torch
from flyinglib import FLYINGLIB_DIRECTORY


class task_config:
    """导航任务的配置"""
    seed = -1
    sim_name = "quad_sim"
    env_name = "env_with_obstacles"
    robot_name = "quadrotor"
    controller_name = "velocity_control"
    args = {}
    num_envs = 2048
    sim_steps = 300
    sim_dt = 0.02
    use_warp = True
    headless = True
    device = "cuda:0"
    observation_space_dim = 17  # position(3) + velocity(3) + orientation(4) + vector_to_target(3) + distance_to_target(1) + quad_angular_velocities(3)
    privileged_observation_space_dim = 0
    action_space_dim = 6  # ax, ay, az, wx, wy, wz

    return_state_before_reset = False  # 通常在重置后为下一个回合返回状态

    # 目标相对于环境边界的位置范围
    target_min_ratio = [0.90, 0.90, 0.90]  # 目标位置相对于环境边界的最小比例 (x,y,z)
    target_max_ratio = [1.2, 1.2, 1.2]  # 目标位置相对于环境边界的最大比例 (x,y,z)

    discount = 0.8 

    # 奖励参数
    reward_parameters = {
        "pos_reward_magnitude": 20.0,           # 位置奖励系数
        "pos_reward_exponent": 1.0 / 3.5,      # 位置奖励指数
        "very_close_to_goal_reward_magnitude": 5.0,  # 非常接近目标的奖励系数
        "very_close_to_goal_reward_exponent": 2.0,   # 非常接近目标的奖励指数
        "getting_closer_reward_multiplier": 10.0,    # 接近目标的进度奖励系数
        "x_action_diff_penalty_magnitude": 0.0,      # x 方向动作差异惩罚系数
        "x_action_diff_penalty_exponent": 3.333,     # x 方向动作差异惩罚指数
        "z_action_diff_penalty_magnitude": 0.0,      # z 方向动作差异惩罚系数
        "z_action_diff_penalty_exponent": 5.0,       # z 方向动作差异惩罚指数
        "yawrate_action_diff_penalty_magnitude": 0.0,  # 偏航速率动作差异惩罚系数
        "yawrate_action_diff_penalty_exponent": 3.33,  # 偏航速率动作差异惩罚指数
        "x_absolute_action_penalty_magnitude": 0.1,    # x 方向动作绝对值惩罚系数
        "x_absolute_action_penalty_exponent": 0.3,     # x 方向动作绝对值惩罚指数
        "z_absolute_action_penalty_magnitude": 1.5,    # z 方向动作绝对值惩罚系数
        "z_absolute_action_penalty_exponent": 1.0,     # z 方向动作绝对值惩罚指数
        "yawrate_absolute_action_penalty_magnitude": 1.5,  # 偏航速率动作绝对值惩罚系数
        "yawrate_absolute_action_penalty_exponent": 2.0,   # 偏航速率动作绝对值惩罚指数
        "collision_penalty": -100.0,                       # 碰撞惩罚
        "crash_penalty": 0,                           # 坠机惩罚
        "attitude_reward_magnitude": 5.0,                  # 姿态奖励系数
        "velocity_reward_magnitude": 3.0,                  # 速度奖励系数
        "thrust_balance_magnitude": 0.0,                   # 推力均衡奖励系数
        "action_magnitude_reward": 0.2,                   # 动作幅度奖励系数
    }

    # 动作转换函数
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

    # 课程学习配置
    class curriculum:
        min_level = 15
        max_level = 50
        check_after_log_instances = 2048
        increase_step = 2
        decrease_step = 1
        success_rate_for_increase = 0.7
        success_rate_for_decrease = 0.6

        def update_curriculim_level(self, success_rate, current_level):
            if success_rate > self.success_rate_for_increase:
                return min(current_level + self.increase_step, self.max_level)
            elif success_rate < self.success_rate_for_decrease:
                return max(current_level - self.decrease_step, self.min_level)
            return current_level 