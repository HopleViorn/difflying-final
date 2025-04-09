# FlyingLib RL 训练框架

此目录包含了用于训练四旋翼飞行器控制策略的强化学习 (RL) 框架。该框架基于 RL Games 库，并集成了 flyinglib 模拟环境。

## 目录结构

```
flyinglib/rl_training/
├── rl_games/               # RL Games 集成
│   ├── runner.py           # 训练脚本
│   ├── ppo_flying_quad.yaml    # 基本 PPO 配置
│   └── ppo_flying_quad_navigation.yaml   # 导航任务 PPO 配置
├── train.py                # 主训练脚本
└── README.md               # 本文档
```

## 安装依赖

确保已安装 RL Games 和其他必要的依赖项：

```bash
pip install rl-games wandb gym
```

## 训练一个策略

要训练四旋翼飞行器执行导航任务，可以使用以下命令：

```bash
python -m flyinglib.rl_training.train --file flyinglib/rl_training/rl_games/ppo_flying_quad_navigation.yaml --task navigation_task --train --num_envs 128 --headless=True
```

### 主要参数

- `--file`: 配置文件路径
- `--task`: 任务名称（例如 `navigation_task`）
- `--train`: 启用训练模式
- `--play`: 启用测试/播放模式（与 `--train` 互斥）
- `--num_envs`: 并行环境数量
- `--headless`: 是否禁用可视化（True/False）
- `--checkpoint`: 加载检查点的路径
- `--seed`: 随机种子
- `--track`: 启用 Weights & Biases 跟踪
- `--wandb-project-name`: W&B 项目名称
- `--wandb-entity`: W&B 团队名称
- `--experiment_name`: 实验名称
- `--rl_device`: RL 算法的设备（例如 cuda:0, cpu）
- `--use_warp`: 是否使用 warp（True/False）

## 自定义任务

要创建自定义任务，需要：

1. 在 `flyinglib/simulation/tasks/` 中创建任务类
2. 在 `flyinglib/simulation/config/` 中创建任务配置
3. 在 `flyinglib/simulation/task_registry.py` 中注册任务
4. 创建适当的 yaml 配置文件

## 奖励函数

默认的导航任务奖励函数包括：

- 位置奖励：接近目标有正奖励
- 接近奖励：朝目标移动有额外奖励
- 动作惩罚：过大或不平滑的动作有惩罚
- 碰撞惩罚：发生碰撞有负奖励

## 示例

### 训练导航策略

```bash
python -m flyinglib.rl_training.train --file flyinglib/rl_training/rl_games/ppo_flying_quad_navigation.yaml --task navigation_task --train --num_envs 128
```

### 测试已训练的策略

```bash
python -m flyinglib.rl_training.train --file flyinglib/rl_training/rl_games/ppo_flying_quad_navigation.yaml --task navigation_task --play --checkpoint runs/flying_navigation_ppo/nn/last_best_flying_navigation_ppo.pth --num_envs 16 --headless=False
```

## 集成 Weights & Biases

要跟踪您的实验，请添加 `--track` 参数：

```bash
python -m flyinglib.rl_training.train --file flyinglib/rl_training/rl_games/ppo_flying_quad_navigation.yaml --task navigation_task --train --num_envs 128 --track --wandb-project-name flyinglib_rl --wandb-entity your_team_name
``` 