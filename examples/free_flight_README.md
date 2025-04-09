# 无人机自由飞行实验

本目录包含用于训练和测试无人机自由飞行控制策略的脚本。

## 文件说明

- `train_free.py`: 训练无人机自由飞行策略的脚本
- `test_free.py`: 测试训练好的策略的脚本
- `run_free_flight.py`: 结合训练和测试的运行脚本

## 使用方法

### 1. 训练策略

运行以下命令训练一个新的策略：

```bash
python examples/train_free.py
```

训练完成后，策略将保存在logs目录下。

### 2. 测试策略

运行以下命令测试一个已经训练好的策略：

```bash
python examples/test_free.py --policy_path <策略文件路径>
```

可选参数：
- `--sim_steps`: 模拟步数，默认为150
- `--sim_dt`: 模拟时间步长，默认为0.02
- `--target_x`, `--target_y`, `--target_z`: 目标位置坐标，默认为(0.7, 0.7, 0.7)

示例：

```bash
python examples/test_free.py --policy_path logs/test/20240710-123456/policy.pth --target_x 1.0 --target_y 0.5 --target_z 0.8
```

### 3. 一键训练和测试

运行以下命令可以一键完成训练和测试：

```bash
python examples/run_free_flight.py
```

可选参数：
- `--epochs`: 训练轮次，默认为500
- `--batch_size`: 批量大小，默认为4096
- `--train_steps`: 训练时的模拟步数，默认为300
- `--test_steps`: 测试时的模拟步数，默认为150
- `--sim_dt`: 模拟时间步长，默认为0.02
- `--target_x`, `--target_y`, `--target_z`: 测试目标位置坐标，默认为(0.7, 0.7, 0.7)
- `--only_test`: 如果提供策略路径，则跳过训练直接测试

示例：

```bash
# 训练并测试
python examples/run_free_flight.py --epochs 200 --target_x 1.0 --target_y 1.0 --target_z 1.0

# 只测试已有策略
python examples/run_free_flight.py --only_test logs/test/20240710-123456/policy.pth
```

## 渲染结果

测试完成后，渲染结果将保存为USD文件，可以使用支持USD格式的软件（如Omniverse）进行查看。

## 参数调整

如需调整训练和测试的参数，可以修改脚本中的相关参数，例如：

- 训练轮次
- 批量大小
- 模拟步数
- 目标位置
- 策略网络结构

通过修改这些参数，可以调整训练效果和测试结果。 