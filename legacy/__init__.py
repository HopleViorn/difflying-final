# flyinglib.simulation 包

# 创建默认模拟器的函数（示例实现）
def create_default_simulator(task_config):
    """创建与任务配置匹配的默认模拟器
    
    Args:
        task_config: 任务配置
        
    Returns:
        创建的模拟器实例
    """
    # 这里只是一个示例，实际实现需要根据 flyinglib 的模拟器 API 来实现
    from flyinglib.simulation.simulator import QuadSimulator
    return QuadSimulator(task_config)

# 导出任务注册表
from .task_registry import task_registry

__all__ = ['create_default_simulator', 'task_registry']
