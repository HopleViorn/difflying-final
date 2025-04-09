import argparse
import os
from examples.train_free import train_free_flight
from examples.test_free import test_free

def run_free_flight():
    """
    训练无人机自由飞行策略并进行测试
    """
    parser = argparse.ArgumentParser(description='训练并测试无人机自由飞行策略')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=4096, help='批量大小')
    parser.add_argument('--train_steps', type=int, default=300, help='训练时的模拟步数')
    parser.add_argument('--test_steps', type=int, default=150, help='测试时的模拟步数')
    parser.add_argument('--sim_dt', type=float, default=0.02, help='模拟时间步长')
    parser.add_argument('--target_x', type=float, default=0.7, help='测试目标位置X坐标')
    parser.add_argument('--target_y', type=float, default=0.7, help='测试目标位置Y坐标')
    parser.add_argument('--target_z', type=float, default=0.7, help='测试目标位置Z坐标')
    parser.add_argument('--only_test', type=str, default=None, help='如果提供策略路径，则跳过训练直接测试')
    
    args = parser.parse_args()
    
    # 设置测试目标位置
    target_pos = [args.target_x, args.target_y, args.target_z]
    
    # 如果提供了策略路径，则跳过训练直接测试
    if args.only_test:
        print(f"跳过训练，直接测试策略: {args.only_test}")
        policy_path = args.only_test
    else:
        # 训练模型
        print("开始训练无人机自由飞行策略...")
        policy_path = train_free_flight(
            epochs=args.epochs,
            batch_size=args.batch_size,
            sim_steps=args.train_steps,
            sim_dt=args.sim_dt
        )
        print(f"训练完成！策略保存在：{policy_path}")
    
    # 测试模型
    print("\n开始测试无人机自由飞行策略...")
    final_pos, final_dist = test_free(
        policy_path=policy_path,
        sim_steps=args.test_steps,
        sim_dt=args.sim_dt,
        target_pos=target_pos
    )
    
    print(f"\n测试完成！")
    print(f"最终位置: {final_pos}")
    print(f"距离目标: {final_dist}")
    
    return policy_path, final_pos, final_dist

if __name__ == "__main__":
    policy_path, final_pos, final_dist = run_free_flight() 