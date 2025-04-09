# flyinglib.simulation.config 包
from flyinglib.simulation.config.navigation_task_config import task_config as navigation_task_config
from flyinglib.simulation.config.hovering_task_config import task_config as hovering_task_config
from flyinglib.simulation.config.forward_flight_task_config import task_config as forward_flight_task_config
from flyinglib.simulation.config.attitude_control_task_config import task_config as attitude_control_task_config

__all__ = ['navigation_task_config', 'hovering_task_config', 'forward_flight_task_config', 'attitude_control_task_config'] 