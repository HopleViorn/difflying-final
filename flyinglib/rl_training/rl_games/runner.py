import numpy as np
import os
import yaml
import torch
import distutils.util
import gym
from gym import spaces
from argparse import Namespace

# 导入 flyinglib 相关模块
from flyinglib import FLYINGLIB_DIRECTORY  # 需要在 __init__.py 中定义这个变量
from flyinglib.simulation import task_registry  # 假设有类似的注册表
from flyinglib.utils.helpers import parse_arguments  # 需要实现这个函数

# 从 rl_games 导入必要的模块
from rl_games.common import env_configurations, vecenv


class ExtractObsWrapper(gym.Wrapper):
    """包装环境以提取观察值"""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        
        # 处理不同的返回格式
        if isinstance(result, tuple):
            # 如果是元组，假设第一个元素是观察值
            observations = result[0]
            if isinstance(observations, dict) and "observations" in observations:
                return observations["observations"]
            return observations
        elif isinstance(result, dict) and "observations" in result:
            # 如果是字典且包含 observations 键
            return result["observations"]
        else:
            # 直接返回结果
            return result

    def step(self, action):
        result = super().step(action)
        
        # 处理不同的返回格式
        if len(result) == 5:  # 新的 gym API (obs, reward, terminated, truncated, info)
            observations, rewards, terminated, truncated, infos = result
            dones = torch.where(
                terminated | truncated,
                torch.ones_like(terminated),
                torch.zeros_like(terminated),
            )
        else:  # 旧的 gym API (obs, reward, done, info)
            observations, rewards, dones, infos = result
        
        # 处理观察值
        if isinstance(observations, dict) and "observations" in observations:
            observations = observations["observations"]
            
        return (
            observations,
            rewards,
            dones,
            infos,
        )


class FlyingRLGPUEnv(vecenv.IVecEnv):
    """飞行机器人 RL 环境的 GPU 实现"""
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.env = ExtractObsWrapper(self.env)
        self.num_actors = num_actors
        
        # 用于记录指标的字典
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = torch.zeros(self.num_actors, device=self.env.task_config.device)
        self.current_lengths = torch.zeros(self.num_actors, device=self.env.task_config.device, dtype=torch.long)

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        
        # 累积奖励和步长
        self.current_rewards += rewards
        self.current_lengths += 1
        
        # 对于已完成的环境，记录并重置
        if torch.any(dones):
            done_indices = torch.where(dones)[0]
            for idx in done_indices:
                self.episode_rewards.append(self.current_rewards[idx].item())
                self.episode_lengths.append(self.current_lengths[idx].item())
                
                # 重置累积奖励和步长
                self.current_rewards[idx] = 0
                self.current_lengths[idx] = 0
        
        # 扩展 infos 以包含更多指标
        if "successes" in infos:
            success_rate = infos["successes"].float().mean().item() if isinstance(infos["successes"], torch.Tensor) else infos["successes"]
            infos["success_rate"] = success_rate
        
        # 添加策略统计信息
        if hasattr(self.env, 'model') and hasattr(self.env.model, 'a2c_network'):
            stddev = self.env.model.a2c_network.logstd.exp().mean().item()
            infos["policy_stddev"] = stddev
        
        # 添加距离和速度信息
        if hasattr(self.env, "sim") and hasattr(self.env.sim, "get_quad_velocities"):
            velocities = self.env.sim.get_quad_velocities()
            infos["velocity"] = torch.norm(velocities, dim=1).mean().item()
            
        if hasattr(self.env, "target_positions") and hasattr(self.env.sim, "get_quad_positions"):
            positions = self.env.sim.get_quad_positions()
            targets = self.env.target_positions
            distances = torch.norm(positions - targets, dim=1)
            infos["distance_to_target"] = distances.mean().item()
        
        # 添加平均奖励和步长
        if len(self.episode_rewards) > 0:
            infos["episode_reward_mean"] = sum(self.episode_rewards[-100:]) / len(self.episode_rewards[-100:])
            infos["episode_length_mean"] = sum(self.episode_lengths[-100:]) / len(self.episode_lengths[-100:])
        
        return obs, rewards, dones, infos

    def reset(self):
        # 重置累积奖励和步长
        self.current_rewards = torch.zeros(self.num_actors, device=self.env.task_config.device)
        self.current_lengths = torch.zeros(self.num_actors, device=self.env.task_config.device, dtype=torch.long)
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = spaces.Box(
            -np.ones(self.env.task_config.action_space_dim),
            np.ones(self.env.task_config.action_space_dim),
        )
        info["observation_space"] = spaces.Box(
            np.ones(self.env.task_config.observation_space_dim) * -np.inf,
            np.ones(self.env.task_config.observation_space_dim) * np.inf,
        )
        print(info["action_space"], info["observation_space"])
        return info
        
    def get_stats(self):
        """返回环境统计信息"""
        stats = {}
        if len(self.episode_rewards) > 0:
            stats["episode_reward_mean"] = sum(self.episode_rewards[-100:]) / len(self.episode_rewards[-100:])
            stats["episode_reward_min"] = min(self.episode_rewards[-100:])
            stats["episode_reward_max"] = max(self.episode_rewards[-100:])
            stats["episode_length_mean"] = sum(self.episode_lengths[-100:]) / len(self.episode_lengths[-100:])
        return stats


# 注册各种环境配置
env_configurations.register(
    "navigation_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("navigation_task", **kwargs),
        "vecenv_type": "FLYING-RLGPU",
    },
)

env_configurations.register(
    "hovering_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("hovering_task", **kwargs),
        "vecenv_type": "FLYING-RLGPU",
    },
)

env_configurations.register(
    "forward_flight_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("forward_flight_task", **kwargs),
        "vecenv_type": "FLYING-RLGPU",
    },
)

env_configurations.register(
    "attitude_control_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("attitude_control_task", **kwargs),
        "vecenv_type": "FLYING-RLGPU",
    },
)

env_configurations.register(
    "position_control_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("position_control_task", **kwargs),
        "vecenv_type": "FLYING-RLGPU",
    },
)

# 注册向量环境类型
vecenv.register(
    "FLYING-RLGPU",
    lambda config_name, num_actors, **kwargs: FlyingRLGPUEnv(config_name, num_actors, **kwargs),
)


def get_args():
    """解析命令行参数"""
    custom_parameters = [
        {
            "name": "--seed",
            "type": int,
            "default": 0,
            "required": False,
            "help": "随机种子，如果大于0将覆盖yaml配置中的值",
        },
        {
            "name": "--train",
            "type": bool,
            "default": True,
            "help": "训练网络",
        },
        {
            "name": "--play",
            "type": bool,
            "default": False,
            "help": "测试网络",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "required": False,
            "help": "检查点路径",
        },
        {
            "name": "--file",
            "type": str,
            "default": "ppo_flying_quad_navigation.yaml",
            "required": False,
            "help": "配置文件路径",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 2048,
            "help": "创建的环境数量。如果提供，将覆盖配置文件",
        },
        {
            "name": "--track",
            "action": "store_true",
            "help": "是否使用Weights and Biases跟踪实验",
        },
        {
            "name": "--wandb-project-name",
            "type": str,
            "default": "flyinglib_rl",
            "help": "wandb项目名称",
        },
        {
            "name": "--wandb-entity",
            "type": str,
            "default": None,
            "help": "wandb团队名称",
        },
        {
            "name": "--task",
            "type": str,
            "default": "navigation_task",
            "help": "如果提供，将覆盖配置文件中的任务",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "default": "flying_navigation_ppo",
            "help": "实验名称。如果提供，将覆盖配置文件",
        },
        {
            "name": "--headless",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "False",
            "help": "强制关闭显示",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "RL算法使用的设备（cpu, gpu, cuda:0, cuda:1等）",
        },
        {
            "name": "--use_warp",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "True",
            "help": "是否使用warp而不是传统渲染管线",
        },
    ]

    # 解析参数
    args = parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # 名称对齐
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def update_config(config, args):
    """更新配置文件"""
    if args.get("task") is not None:
        config["params"]["config"]["env_name"] = args["task"]
    if args.get("experiment_name") is not None:
        config["params"]["config"]["name"] = args["experiment_name"]
        
    # 确保 env_config 存在
    if "env_config" not in config["params"]["config"]:
        config["params"]["config"]["env_config"] = {}
        
    config["params"]["config"]["env_config"]["headless"] = args.get("headless", False)
    config["params"]["config"]["env_config"]["num_envs"] = args.get("num_envs", 1024)
    config["params"]["config"]["env_config"]["use_warp"] = args.get("use_warp", True)
    
    if args.get("num_envs", 0) > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    if args.get("seed", 0) > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]

    config["params"]["config"]["player"] = {"use_vecenv": True}
    
    # 设置训练和播放模式
    config["params"]["train"] = args.get("train", True)
    config["params"]["play"] = args.get("play", False)
    
    # 如果提供了检查点，则设置加载检查点
    if args.get("checkpoint") is not None:
        config["params"]["load_checkpoint"] = True
        config["params"]["load_path"] = args["checkpoint"]
    
    # 确保日志记录间隔设置正确
    if "log_interval" not in config["params"]:
        config["params"]["log_interval"] = 10  # 设置日志记录间隔
    
    # 添加额外指标记录
    config["params"]["config"]["stats_to_log"] = [
        "reward", 
        "episode_length", 
        "velocity", 
        "distance_to_target", 
        "success_rate"
    ]
    
    # 添加回调函数配置
    if "tensorboard" not in config["params"]:
        config["params"]["tensorboard"] = True
    
    # 确保 wandb 配置正确
    if args.get("track", False):
        if "wandb" not in config["params"]:
            config["params"]["wandb"] = {}
        
        config["params"]["wandb"]["project"] = args.get("wandb_project_name", "flyinglib-rl")
        config["params"]["wandb"]["group"] = args.get("experiment_name", args.get("task", "default"))
        config["params"]["wandb"]["log_interval"] = 10
    
    return config


if __name__ == "__main__":
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(get_args())

    config_name = args["file"]

    print("加载配置: ", config_name)
    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)

        config = update_config(config, args)

        from rl_games.torch_runner import Runner

        runner = Runner()
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if args.get("track", False) and rank == 0:
        import wandb

        wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )
    runner.run(args)

    if args.get("track", False) and rank == 0:
        wandb.finish()