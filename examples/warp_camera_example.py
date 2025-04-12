import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from flyinglib.sensors.warp.warp_cam import WarpCam
import torch

# Initialize Warp
wp.init()

# Create a simple box mesh
def create_box_mesh(size=1.0):
    vertices = np.array([
        [-size, -size, -size],
        [size, -size, -size], 
        [size, size, -size],
        [-size, size, -size],
        [-size, -size, size],
        [size, -size, size],
        [size, size, size],
        [-size, size, size]
    ], dtype=np.float32)

    faces = np.array([
        [0,1,2], [0,2,3],  # bottom
        [4,5,6], [4,6,7],  # top
        [0,1,5], [0,5,4],  # front
        [2,3,7], [2,7,6],  # back
        [1,2,6], [1,6,5],  # right
        [0,3,7], [0,7,4]   # left
    ], dtype=np.int32)

    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3),
        indices=wp.array(faces.flatten(), dtype=wp.int32),
        velocities=wp.zeros(len(vertices), dtype=wp.vec3)
    )
    return mesh

def create_sphere_mesh(radius=1.0):
    # Create a unit sphere mesh
    subdivisions = 16  # Number of subdivisions for sphere approximation
    
    # Generate vertices
    vertices = []
    for i in range(subdivisions + 1):
        theta = i * np.pi / subdivisions  # Polar angle
        for j in range(subdivisions):
            phi = j * 2 * np.pi / subdivisions  # Azimuthal angle
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            vertices.append([x, y, z])
    
    # Generate faces
    faces = []
    for i in range(subdivisions):
        for j in range(subdivisions):
            p1 = i * subdivisions + j
            p2 = i * subdivisions + (j + 1) % subdivisions
            p3 = (i + 1) * subdivisions + j
            p4 = (i + 1) * subdivisions + (j + 1) % subdivisions
            
            if i != 0:  # Bottom cap
                faces.append([p1, p2, p3])
            if i != subdivisions - 1:  # Top cap
                faces.append([p3, p2, p4])
    
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    
    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3),
        indices=wp.array(faces.flatten(), dtype=wp.int32),
        velocities=wp.zeros(len(vertices), dtype=wp.vec3)
    )
    return mesh

class CameraConfig:
    def __init__(self, config_dict):
        self.width = config_dict['width']
        self.height = config_dict['height']
        self.horizontal_fov_deg = config_dict['horizontal_fov_deg']
        self.max_range = config_dict['max_range']
        self.calculate_depth = config_dict['calculate_depth']
        self.num_sensors = config_dict['num_sensors']
        self.segmentation_camera = config_dict['segmentation_camera']
        self.return_pointcloud = config_dict['return_pointcloud']

def main():
    # Create config
    cfg = CameraConfig(
        config_dict={
            'width': 640,
            'height': 480,
            'horizontal_fov_deg': 60.0,
            'max_range': 100.0,
            'calculate_depth': True,
            'num_sensors': 1,
            'segmentation_camera': False,
            'return_pointcloud': False
        }
    )
    
    # Create room (large box)
    room_size = 5.0
    room_mesh = create_box_mesh(size=room_size)
    
    # Create random objects
    num_objects = 10
    meshes = [room_mesh]
    
    for _ in range(num_objects):
        # Randomly choose between box or sphere
        if np.random.rand() > 0.5:
            obj = create_box_mesh(size=np.random.uniform(0.2, 0.5))
        else:
            obj = create_sphere_mesh(radius=np.random.uniform(0.2, 0.5))
        
        # Random position inside room (with margin)
        pos = np.random.uniform(-room_size*0.8, room_size*0.8, size=3)
        obj.points = wp.array(obj.points.numpy() + pos, dtype=wp.vec3)
        meshes.append(obj)
    
    # Combine all mesh IDs
    mesh_ids = wp.array([m.id for m in meshes], dtype=wp.uint64)
    
    # Create camera
    camera = WarpCam(num_envs=1, config=cfg, mesh_ids_array=mesh_ids)
    
    # 将摄像机放在更靠近房间中心的位置，并面向房间内部
    # 放在一个角落，但在房间内
    corner_pos = np.array([room_size * 0.5, room_size * 0.5, room_size * 0.5])
    # Reshape to [num_envs, num_sensors, 3]
    cam_pos = torch.tensor([[[corner_pos[0], corner_pos[1], corner_pos[2]]]], device='cuda:0', dtype=torch.float32)
    
    # 摄像机朝向房间中心(0,0,0)的方向
    center = np.array([0.0, 0.0, 0.0])
    forward = center - corner_pos  # 从摄像机位置指向中心
    forward = forward / np.linalg.norm(forward)  # 归一化
    
    # 计算右方向（可以用任意一个与forward不共线的向量计算叉积）
    up_world = np.array([0.0, 0.0, 1.0])  # 世界坐标系的上方向
    right = np.cross(forward, up_world)
    right = right / np.linalg.norm(right)  # 归一化
    
    # 计算真正的上方向
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)  # 归一化
    
    # 构建旋转矩阵
    rot_matrix = np.eye(3)
    rot_matrix[:,0] = right
    rot_matrix[:,1] = up
    rot_matrix[:,2] = forward
    
    # Convert rotation matrix to quaternion
    trace = rot_matrix[0,0] + rot_matrix[1,1] + rot_matrix[2,2]
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (rot_matrix[2,1] - rot_matrix[1,2]) / S
        qy = (rot_matrix[0,2] - rot_matrix[2,0]) / S
        qz = (rot_matrix[1,0] - rot_matrix[0,1]) / S
    elif (rot_matrix[0,0] > rot_matrix[1,1]) and (rot_matrix[0,0] > rot_matrix[2,2]):
        S = np.sqrt(1.0 + rot_matrix[0,0] - rot_matrix[1,1] - rot_matrix[2,2]) * 2
        qw = (rot_matrix[2,1] - rot_matrix[1,2]) / S
        qx = 0.25 * S
        qy = (rot_matrix[0,1] + rot_matrix[1,0]) / S
        qz = (rot_matrix[0,2] + rot_matrix[2,0]) / S
    elif (rot_matrix[1,1] > rot_matrix[2,2]):
        S = np.sqrt(1.0 + rot_matrix[1,1] - rot_matrix[0,0] - rot_matrix[2,2]) * 2
        qw = (rot_matrix[0,2] - rot_matrix[2,0]) / S
        qx = (rot_matrix[0,1] + rot_matrix[1,0]) / S
        qy = 0.25 * S
        qz = (rot_matrix[1,2] + rot_matrix[2,1]) / S
    else:
        S = np.sqrt(1.0 + rot_matrix[2,2] - rot_matrix[0,0] - rot_matrix[1,1]) * 2
        qw = (rot_matrix[1,0] - rot_matrix[0,1]) / S
        qx = (rot_matrix[0,2] + rot_matrix[2,0]) / S
        qy = (rot_matrix[1,2] + rot_matrix[2,1]) / S
        qz = 0.25 * S
    
    # Reshape to [num_envs, num_sensors, 4]
    cam_quat = torch.tensor([[[qx, qy, qz, qw]]], device='cuda:0', dtype=torch.float32)
    
    # Set tensors
    depth_image = torch.zeros((1, 1, cfg.height, cfg.width), device='cuda:0')
    camera.set_image_tensors(depth_image)
    # Reshape to ensure correct dimensions [num_envs, num_sensors, dim]
    camera.set_pose_tensor(
        cam_pos.reshape(1, 1, 3), 
        cam_quat.reshape(1, 1, 4)
    )
    
    # Render
    camera.capture()

    # Get and show result
    depth = depth_image.cpu().numpy()[0,0]

    # 创建图像并设置样式
    plt.figure(figsize=(10, 8))
    im = plt.imshow(depth, cmap='viridis')
    plt.colorbar(im, label='Depth (m)')
    
    # 添加坐标轴标签
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.title('Depth Image')

    # 保存图像
    plt.savefig('depth_image.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Depth image saved as depth_image.png")

    
    # 改变相机位置
    cam_pos = torch.tensor([[[0.0, 0.0, 3.0]]], device='cuda:0', dtype=torch.float32)
    cam_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device='cuda:0', dtype=torch.float32)
    camera.set_pose_tensor(
        cam_pos.reshape(1, 1, 3), 
        cam_quat.reshape(1, 1, 4)
    )

    camera.capture()

    # Get and show result
    depth = depth_image.cpu().numpy()[0,0]

    # 创建图像并设置样式
    plt.figure(figsize=(10, 8))
    im = plt.imshow(depth, cmap='viridis')
    plt.colorbar(im, label='Depth (m)')
    
    # 添加坐标轴标签
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.title('Depth Image')

    # 保存图像
    plt.savefig('depth_image-2.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Depth image saved as depth_image.png")
    
  

if __name__ == "__main__":
    main()