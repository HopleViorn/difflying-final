import numpy as np
import torch
import torch.nn as nn
import argparse
from flyinglib.control.quad_lee_controller import QuadLeeController
import datetime
import os
import matplotlib.pyplot as plt
from PIL import Image
import imageio

from flyinglib.objects.propeller import *
from flyinglib.objects.drone import *
from flyinglib.simulation.step import *
from flyinglib.scene.scene_manager import SceneManager
import cv2

from policy_nn import GRUPolicyNetwork
from policy_nn import img_size


DEVICE = "cuda:0"
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def action_transformation_function(action):
    #限制动作范围
    processed_action = action
    processed_action[:, 0:3] = processed_action[:, 0:3] * 0.5
    processed_action[:, 3:] = processed_action[:, 3:] * 0.0001
    return processed_action


def test_free(
    policy_path: str,
    sim_steps: int = 200,
    sim_dt: float = 0.04,
    target_pos = None
):
    """
    测试训练好的策略网络并渲染结果
    
    Args:
        policy_path: 策略网络权重的路径
        sim_steps: 模拟步数
        sim_dt: 模拟时间步长
        target_pos: 目标位置（仅用于可视化）
    """
    print(f"加载策略：{policy_path}")
    
    # 创建单个无人机实例
    drone = Drone('test', batch_size=1, sim_steps=sim_steps, sim_dt=sim_dt)
    
    # 初始化位置和姿态
    q = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], device=DEVICE)
    qd = torch.zeros((1, 6), device=DEVICE)
    
    # 设置目标位置（仅用于可视化）
    if target_pos is None:
        target_pos = [2, 1.5, 0]
    target_pos_tensor = torch.tensor([target_pos], device=DEVICE)
    
    # 加载GRU策略网络
    policy = GRUPolicyNetwork(input_dim=10, output_dim=6, hidden_size=128)
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()
    
    # 创建控制器
    controller = QuadLeeController(num_envs=1, device=DEVICE, drone=drone)
    
    # 初始化环境管理器并设置场景
    env_manager = SceneManager(batch_size=1)
    env_manager.setup_room()
    
    # 添加前置摄像头（使用训练时的图像尺寸）
    camera_config = {
        'width': img_size[0],
        'height': img_size[1],
        'horizontal_fov_deg': 120,
        'max_range': 20.0,
        'calculate_depth': True,
        'num_sensors': 1,
        'segmentation_camera': False,
        'return_pointcloud': False
    }
    env_manager.add_camera('front', config=camera_config, positions=q[:, :3], orientations=q[:, 3:])
    
    # 准备保存深度图和渲染图像
    depth_images = []
    output_dir = f"outputs/test_free_depth_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"开始模拟")
    
    hidden_state = None
    
    # 模拟并渲染每一步
    for step in range(sim_steps):
        pos = q[:, :3]  # 位置
        att = q[:, 3:]  # 姿态
        angular_vel = qd[:, :3]  # 角速度
        vel = qd[:, 3:]  # 速度
        print(torch.norm(vel, dim=1))
        
        # 更新相机位置
        env_manager.set_camera_pose_tensor('front', pos, att)
        depth_img = env_manager.capture_depth('front')
        depth_img = depth_img.squeeze(0)
        
        # 构建观察向量（匹配训练配置）
        obs = torch.cat([
            vel,  # 速度 (3)
            angular_vel,  # 角速度 (3)
            att,  # 姿态 (4)
        ], dim=1).to(DEVICE)
        
        # 使用策略网络生成动作
        with torch.no_grad():
            a, hidden_state = policy(obs, depth_img, hidden_state)
            a = action_transformation_function(a)
            a = controller.accelerations_to_motor_thrusts(a[:, :3], a[:, 3:], att)
        
        # 模拟一步
        q, qd = diff_step(q, qd, a, drone)
        
        # 保存深度图（每5步）
        if step % 5 == 0:
            depth_img_np = depth_img.squeeze().cpu().numpy()
            plt.figure(figsize=(4, 4))
            depth_img_np = cv2.rotate(depth_img_np, cv2.ROTATE_180)
            plt.imshow(depth_img_np, cmap='viridis')
            plt.colorbar(label='Depth')
            pos_str = f"pos: [{pos[0,0]:.2f}, {pos[0,1]:.2f}, {pos[0,2]:.2f}]"
            att_str = f"att: [{att[0,0]:.2f}, {att[0,1]:.2f}, {att[0,2]:.2f}, {att[0,3]:.2f}]"
            plt.title(f"{pos_str}\n{att_str}", fontsize=8)
            plt.axis('off')
            depth_filename = os.path.join(output_dir, f'depth_{step:03d}.png')
            plt.savefig(depth_filename, bbox_inches='tight', pad_inches=0)
            plt.close()
            depth_images.append(depth_filename)
        
        # 渲染当前状态
        obstacles = [{"position": torch.tensor(obj["position"]).cpu().numpy(),
                     "radius": obj["radius"],
                     "type": obj["type"]} for obj in env_manager.objects]
        drone.render(target_pos, obstacles)
        
        # 打印状态（每30步）
        if step % 30 == 0:
            current_pos = pos.detach().cpu().numpy()[0]
            dist = torch.norm(target_pos_tensor - pos, dim=1).item()
            print(f"步骤 {step}: 位置 {current_pos}, 距离目标 {dist:.4f}")
    
    # 打印最终结果
    final_pos = q[:, :3].detach().cpu().numpy()[0]
    final_dist = torch.norm(target_pos_tensor - q[:, :3], dim=1).item()
    
    print(f"\n=== 模拟完成 ===")
    print(f"最终位置: {final_pos}")
    print(f"距离目标: {final_dist}")
    print(f"最终姿态: {q.detach().cpu().numpy()[0]}")
    print(f"最终速度: {qd.detach().cpu().numpy()[0]}")
    
    # 保存渲染结果和GIF
    drone.renderer.save()
    if depth_images:
        create_gif(depth_images, os.path.join(output_dir, 'depth.gif'))
    
    return final_pos, final_dist

def create_gif(image_files, output_file, duration=0.1):
    """
    从图像文件列表创建GIF
    
    Args:
        image_files: 图像文件路径列表
        output_file: 输出GIF文件路径
        duration: 每帧持续时间（秒）
    """
    import cv2
    images = []
    for filename in image_files:
        # Read image and ensure RGB format
        image = imageio.imread(filename)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Resize image to 330x330
        image = cv2.resize(image, (330, 330))
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        images.append(image)

    # Save as GIF with proper color handling
    imageio.mimsave(output_file, images, duration=duration, format='GIF')

    # Delete input images
    for filename in image_files:
        os.remove(filename)
    
    print(f"GIF已保存到: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试训练好的无人机自由飞行策略')
    parser.add_argument('--path', type=str, required=True, help='策略网络权重的路径')
    parser.add_argument('--sim_steps', type=int, default=200, help='模拟步数')
    parser.add_argument('--sim_dt', type=float, default=0.06, help='模拟时间步长')
    
    args = parser.parse_args()
    
    final_pos, final_dist = test_free(
        policy_path=args.path,
        sim_steps=args.sim_steps,
        sim_dt=args.sim_dt,
        target_pos = [1.5, 0.7, 0]
    )
    
    print(f"\n测试完成！最终位置: {final_pos}, 距离目标: {final_dist}")
