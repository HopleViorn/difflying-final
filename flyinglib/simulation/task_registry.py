import torch
from typing import Dict, Any, Optional, Callable, Type

# 导入各种任务
from flyinglib.simulation.tasks import NavigationTask, HoveringTask, ForwardFlightTask, AttitudeControlTask
from flyinglib.simulation.config import navigation_task_config, hovering_task_config, forward_flight_task_config, attitude_control_task_config


class TaskRegistry:
    """任务注册表，用于注册和创建各种任务"""
    
    def __init__(self):
        self.task_map = {}
        self.config_map = {}
        self.sim_map = {}
        
        # 注册内置任务
        self._register_default_tasks()
    
    def _register_default_tasks(self):
        """注册默认任务"""
        # 注册导航任务
        self.register_task(
            name="navigation_task",
            task_class=NavigationTask,
            task_config=navigation_task_config,
            sim_creator=None  # 使用默认模拟器创建函数
        )
        
        # 注册悬停任务
        self.register_task(
            name="hovering_task",
            task_class=HoveringTask,
            task_config=hovering_task_config,
            sim_creator=None
        )
        
        # 注册前进飞行任务
        self.register_task(
            name="forward_flight_task",
            task_class=ForwardFlightTask,
            task_config=forward_flight_task_config,
            sim_creator=None
        )
        
        # 注册姿态控制任务
        self.register_task(
            name="attitude_control_task",
            task_class=AttitudeControlTask,
            task_config=attitude_control_task_config,
            sim_creator=None
        )

    def register_task(self, name: str, task_class: Type, task_config: Any, sim_creator: Optional[Callable] = None):
        """注册一个新任务
        
        Args:
            name: 任务名称
            task_class: 任务类
            task_config: 任务配置
            sim_creator: 模拟器创建函数，如果为 None，则使用默认创建函数
        """
        self.task_map[name] = task_class
        self.config_map[name] = task_config
        self.sim_map[name] = sim_creator
    
    def make_task(self, name: str, **kwargs):
        """创建一个任务实例
        
        Args:
            name: 任务名称
            **kwargs: 传递给模拟器创建函数的参数
        
        Returns:
            创建的任务实例
        """
        if name not in self.task_map:
            raise ValueError(f"未知任务名称：{name}")
        
        # 获取任务类和配置
        task_class = self.task_map[name]
        task_config = self.config_map[name]
        
        # 处理可能被覆盖的配置参数
        for key, value in kwargs.items():
            if hasattr(task_config, key):
                setattr(task_config, key, value)
        
        # 设置配置中的设备
        if hasattr(kwargs, 'rl_device') and hasattr(task_config, 'device'):
            task_config.device = kwargs['rl_device']
        
        # 创建模拟器
        sim_creator = self.sim_map[name]
        if sim_creator is None:
            from flyinglib.simulation import create_default_simulator
            sim = create_default_simulator(task_config)
        else:
            sim = sim_creator(task_config)
        
        # 创建任务实例
        task = task_class(sim, task_config)
        
        # 添加必要的属性
        if not hasattr(task, 'task_config'):
            task.task_config = task_config
            
        # 确保任务有 observation_space_dim 和 action_space_dim 属性
        if not hasattr(task, 'observation_space_dim') and hasattr(task_config, 'observation_space_dim'):
            task.observation_space_dim = task_config.observation_space_dim
            
        if not hasattr(task, 'action_space_dim') and hasattr(task_config, 'action_space_dim'):
            task.action_space_dim = task_config.action_space_dim
            
        return task


# 创建全局任务注册表实例
task_registry = TaskRegistry() 