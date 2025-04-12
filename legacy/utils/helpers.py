import argparse
import os
import random
import numpy as np
import torch


def set_seed(seed, torch_deterministic=False):
    """设置随机种子，使实验可重现"""
    if seed == -1:
        seed = random.randint(0, 10000)
    print(f"设置随机种子: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch_deterministic:
        # 设置确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 启用 cudnn 自动调优
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    return seed


def parse_arguments(description="飞行机器人 RL 训练", custom_parameters=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description=description)
    
    # 添加基本参数
    parser.add_argument('--compute_device_id', type=int, default=0,
                        help='设备 ID 用于 RL 训练 (默认: 0)')
    parser.add_argument('--sim_device_id', type=int, default=0,
                        help='设备 ID 用于物理模拟 (默认: 0)')
    parser.add_argument('--compute_device_type', type=str, default='cuda',
                        help='计算设备类型 (cuda 或 cpu)')
    parser.add_argument('--sim_device_type', type=str, default='cuda',
                        help='模拟设备类型 (cuda 或 cpu)')
    
    # 添加自定义参数
    if custom_parameters is not None:
        for param in custom_parameters:
            if 'name' in param and 'type' in param and 'default' in param and 'help' in param:
                if 'action' in param:
                    parser.add_argument(
                        param['name'], 
                        type=param['type'], 
                        default=param['default'],
                        help=param['help'],
                        action=param['action']
                    )
                else:
                    parser.add_argument(
                        param['name'], 
                        type=param['type'], 
                        default=param['default'],
                        help=param['help']
                    )
    
    args = parser.parse_args()
    return args


def make_dir(path):
    """创建目录（如果不存在）"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path 