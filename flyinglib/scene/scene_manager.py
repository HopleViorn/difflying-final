import warp as wp
import numpy as np
import torch
from typing import List, Dict, Optional, Union
from flyinglib.sensors.warp.warp_cam import WarpCam


class CameraConfig:
    def __init__(self, config_dict: Dict):
        self.width = config_dict['width']
        self.height = config_dict['height']
        self.horizontal_fov_deg = config_dict['horizontal_fov_deg']
        self.max_range = config_dict['max_range']
        self.calculate_depth = config_dict['calculate_depth']
        self.num_sensors = config_dict['num_sensors']
        self.segmentation_camera = config_dict['segmentation_camera']
        self.return_pointcloud = config_dict['return_pointcloud']

class SceneManager:
    def __init__(self, batch_size: int = 1):
        """Initialize the scene manager with batch support
        
        Args:
            batch_size: Number of parallel environments/simulations
        """
        wp.init()
        self.batch_size = batch_size
        self.meshes = []  # List of warp meshes in the scene
        self.cameras: Dict[str, WarpCam] = {}  # Dict of cameras: {camera_id: WarpCam}
        self.mesh_ids = None  # Combined mesh IDs array
        self.depth_image = None
        self.objects = []  # List of objects in scene: [{"position": vec3, "radius": float}]

    def create_box_mesh(self, size=1.0, flip_normals=False):
        """Create a box mesh with given size
        
        Args:
            size: Size of the box
            flip_normals: Whether to flip face normals (reverse winding order)
        """
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

        if flip_normals:
            # Reverse winding order to flip normals
            faces = np.flip(faces, axis=1)

        mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3),
            indices=wp.array(faces.flatten(), dtype=wp.int32),
            velocities=wp.zeros(len(vertices), dtype=wp.vec3)
        )
        return mesh

    def create_sphere_mesh(self, radius=1.0, flip_normals=False):
        """Create a sphere mesh with given radius
        
        Args:
            radius: Radius of the sphere
            flip_normals: Whether to flip face normals (reverse winding order)
        """
        subdivisions = 16
        vertices = []
        for i in range(subdivisions + 1):
            theta = i * np.pi / subdivisions
            for j in range(subdivisions):
                phi = j * 2 * np.pi / subdivisions
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)
                vertices.append([x, y, z])
        
        faces = []
        for i in range(subdivisions):
            for j in range(subdivisions):
                p1 = i * subdivisions + j
                p2 = i * subdivisions + (j + 1) % subdivisions
                p3 = (i + 1) * subdivisions + j
                p4 = (i + 1) * subdivisions + (j + 1) % subdivisions
                
                if i != 0:
                    faces.append([p1, p2, p3])
                if i != subdivisions - 1:
                    faces.append([p3, p2, p4])
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)

        if flip_normals:
            # Reverse winding order to flip normals
            faces = np.flip(faces, axis=1)
        
        mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3),
            indices=wp.array(faces.flatten(), dtype=wp.int32),
            velocities=wp.zeros(len(vertices), dtype=wp.vec3)
        )
        return mesh

    def setup_room(self, room_size=5.0, num_objects=5):
        """Setup a room with random obstacles
        
        Args:
            room_size: Size of the room (half-extent)
            num_objects: Number of random objects to add
        """
        # Clear existing objects
        self.objects.clear()
        
        # Initialize combined mesh data
        all_points = []
        all_indices = []
        all_velocities = []
        vertex_count = 0

        # Create room and add to combined mesh
        room = self.create_box_mesh(size=room_size, flip_normals=True)
        room_points = room.points.numpy()
        room_indices = room.indices.numpy()
        room_velocities = room.velocities.numpy()
        
        all_points.extend(room_points)
        all_indices.extend(room_indices)
        all_velocities.extend(room_velocities)
        vertex_count += len(room_points)

        # Add random objects
        for i in range(num_objects):
            if np.random.rand() > 0.5:
                size = np.random.uniform(0.1, 0.2)
                obj = self.create_box_mesh(size=size, flip_normals=True)
                radius = size * 0.5  # Approximate box with sphere
                obj_type = "box"
            else:
                radius = np.random.uniform(0.1, 0.2)
                obj = self.create_sphere_mesh(radius=radius, flip_normals=True)
                obj_type = "sphere"
            # Random position (avoid center)
            # pos = np.random.uniform(0.5, 1.2, size=3)

            # # 在球面上均匀采样点
            # # 使用高斯分布生成三维向量
            # pos = np.random.normal(0, 1, size=3)
            # # 归一化到单位球面
            # pos = pos / np.linalg.norm(pos)
            
            # # 缩放到所需距离(在0.5到room_size-0.5之间)
            # distance = np.random.uniform(0.6, 0.7)
            # pos = pos * distance + 1.2

            #分布在距离原点一定距离以外的空间
            # 生成随机角度
            # theta = np.random.uniform(0, 2 * np.pi)
            # phi = np.random.uniform(0, np.pi)
            
            # # 将球坐标转换为笛卡尔坐标
            # x = np.sin(phi) * np.cos(theta)
            # y = np.sin(phi) * np.sin(theta)
            # z = np.cos(phi)
            # pos = np.array([x, y, z])

            # pos = np.random.normal(0, 1, size=3)
            
            # # 归一化并缩放到所需距离
            # pos = pos / np.linalg.norm(pos)
            # distance = np.random.uniform(0.6, 1)  # 控制障碍物距离原点的范围
            # pos = pos * distance

            x = np.random.uniform(-0.1, 0.1)
            y = np.random.uniform(-2, 2)
            z = np.random.uniform(-0.2, 0.2) + 0.7

            pos = np.array([x, y, z])

            obj_points = obj.points.numpy() + pos
            obj_indices = obj.indices.numpy() + vertex_count
            obj_velocities = obj.velocities.numpy()
            
            # Store object info for collision detection
            self.objects.append({
                "position": pos,
                "radius": radius,
                "type": obj_type
            })
            
            all_points.extend(obj_points)
            all_indices.extend(obj_indices)
            all_velocities.extend(obj_velocities)
            vertex_count += len(obj_points)
            
            # print(f"Object {i} position: {pos}, vertices: {len(obj_points)}, faces: {len(obj_indices)//3}")

        # Create single combined mesh
        combined_mesh = wp.Mesh(
            points=wp.array(all_points, dtype=wp.vec3),
            indices=wp.array(all_indices, dtype=wp.int32),
            velocities=wp.array(all_velocities, dtype=wp.vec3)
        )
        self.add_mesh(combined_mesh)
        # print(f"Created combined mesh with {len(all_points)} vertices and {len(all_indices)//3} faces")
        
    def add_mesh(self, mesh: wp.Mesh) -> None:
        """Add a mesh to the scene
        
        Args:
            mesh: Warp mesh object to add to the scene
        """
        self.meshes.append(mesh)
        self._update_mesh_ids()
        
    def remove_mesh(self, mesh: wp.Mesh) -> None:
        """Remove a mesh from the scene
        
        Args:
            mesh: Warp mesh object to remove
        """
        if mesh in self.meshes:
            self.meshes.remove(mesh)
            self._update_mesh_ids()
            
    def _update_mesh_ids(self) -> None:
        """Update the combined mesh IDs array"""
        # print(f"Updating mesh_ids with {len(self.meshes)} meshes")
        if self.meshes:
            self.mesh_ids = wp.array([m.id for m in self.meshes], dtype=wp.uint64)
        else:
            self.mesh_ids = None
            
    def add_camera(self, 
                  camera_id: str,
                  config: Dict,
                  positions: torch.Tensor,
                  orientations: torch.Tensor) -> None:
        """Add a camera to the scene with batch support
        
        Args:
            camera_id: Unique identifier for the camera
            config: Camera configuration dictionary
            positions: Camera positions [batch_size, 3] 
            orientations: Camera orientations as quaternions [batch_size, 4]
        """
        if self.mesh_ids is None:
            raise ValueError("No meshes in scene, cannot add camera")
        
        self.depth_image = torch.zeros(
            (1, self.batch_size, config['height'], config['width']),
            device='cuda:0',
            requires_grad=False
        )
            
        # Create camera with batch support
        camera = WarpCam(
            num_envs = 1,
            config=CameraConfig(config),
            mesh_ids_array=self.mesh_ids
        )
        
        
        camera.set_pose_tensor(positions, orientations)
        camera.set_image_tensors(self.depth_image)

        self.cameras[camera_id] = camera

    def set_camera_pose_tensor(self,
                           camera_id: str,
                           positions: torch.Tensor,
                           orientations: torch.Tensor,
                           ) -> None:
        """Set camera pose with optional predefined directions
        
        Args:
            camera_id: ID of camera to set pose for
            positions: Camera positions [batch_size, 3]
            orientations: Base orientations as quaternions [batch_size, 4]
            direction: Optional direction to override orientation:
                'up', 'down', 'left', 'right', 'front', 'back'
        """
        direction = 'up'
        # Create rotation quaternions for each direction relative to base orientation
        # All quaternions are normalized
        direction_quats = {
            'up': torch.tensor([0.0, 0.0, 0.0, 1.0]),  # Same as base
            'down': torch.tensor([1.0, 0.0, 0.0, 0.0]),  # 180° around x
            'left': torch.tensor([0.0, 0.0, 0.7071, 0.7071]),  # 90° around y
            'right': torch.tensor([0.0, 0.0, -0.7071, 0.7071]),  # -90° around y
            'front': torch.tensor([0.7071, 0.0, 0.0, 0.7071]),  # 90° around x
            'back': torch.tensor([-0.7071, 0.0, 0.0, 0.7071])  # -90° around x
        }
        
        # Get the direction quaternion and ensure same device as orientations
        dir_quat = direction_quats[direction].to(orientations.device)
        
        w = orientations[..., 3:4] * dir_quat[3] - torch.sum(orientations[..., :3] * dir_quat[:3], dim=-1, keepdim=True)
        xyz = orientations[..., 3:4] * dir_quat[:3] + dir_quat[3] * orientations[..., :3] + torch.cross(orientations[..., :3], dir_quat[:3].expand_as(orientations[..., :3]))
        orientations = torch.cat([xyz, w], dim=-1)
        
        # Normalize the resulting quaternion
        orientations = orientations / torch.norm(orientations, dim=-1, keepdim=True)

        self.cameras[camera_id].set_pose_tensor(
            positions.reshape(1, self.batch_size, 3),
            orientations.reshape(1, self.batch_size, 4)
        )
        
    def remove_camera(self, camera_id: str) -> None:
        """Remove a camera from the scene
        
        Args:
            camera_id: ID of camera to remove
        """
        self.cameras.pop(camera_id, None)
        
    def capture_depth(self, camera_id: str) -> torch.Tensor:
        """Capture depth images from specified camera for all batches
        
        Args:
            camera_id: ID of camera to capture from
            
        Returns:
            Depth images as torch tensor [batch_size, height, width]
        """
        if camera_id not in self.cameras:
            raise ValueError(f"Camera {camera_id} not found")
            
        camera = self.cameras[camera_id]
        return camera.capture()
        
    def get_nearest_object_distance(self, position: torch.Tensor, temperature: float = 0.1) -> tuple:
        """Calculate differentiable distance vector to nearest object's collision sphere
        
        Args:
            position: 3D position(s) to check from [..., 3] or [batch_size, ..., 3]
            temperature: Softmax temperature for softmin operation (lower = sharper)
            
        Returns:
            tuple: (distance_vector, distance_magnitude) to nearest object
        """
        if not self.objects:
            return (
                torch.zeros_like(position, dtype=torch.float32),
                torch.zeros(
                    position.shape[:-1],
                    device=position.device,
                    dtype=torch.float32,
                    requires_grad=True
                )
            )
            
        # Convert objects to tensors with matching dtype and device
        obj_positions = torch.stack([
            torch.as_tensor(obj["position"],
                          device=position.device,
                          dtype=torch.float32)
            for obj in self.objects
        ])  # [num_objects, 3]
        
        obj_radii = torch.as_tensor(
            [obj["radius"] for obj in self.objects],
            device=position.device,
            dtype=torch.float32
        )  # [num_objects]
        
        # Ensure position is float32 if it isn't already
        position = position.to(dtype=torch.float32)
        
        # Calculate vectors from positions to objects [..., num_objects, 3]
        vecs = obj_positions - position.unsqueeze(-2)
        
        # Calculate distances [..., num_objects]
        dists = torch.norm(vecs, dim=-1) - obj_radii
        
        # Softmin weights [..., num_objects]
        weights = torch.nn.functional.softmin(dists / temperature, dim=-1)
        
        # Weighted average of distances and vectors
        nearest_dists = torch.sum(weights * dists, dim=-1)
        nearest_vecs = torch.sum(weights.unsqueeze(-1) * vecs, dim=-2)
        # Normalize vectors and scale by distance where distance > 0
        mask = nearest_dists > 0
        if mask.any():
            # Create new tensor instead of inplace modification
            norm_vecs = nearest_vecs / (torch.norm(nearest_vecs, dim=-1, keepdim=True) + 1e-8)
            scaled_vecs = norm_vecs * nearest_dists.unsqueeze(-1)
            nearest_vecs = torch.where(mask.unsqueeze(-1), scaled_vecs, nearest_vecs)
            nearest_vecs[mask] = norm_vecs[mask] * nearest_dists[mask].unsqueeze(-1)
            
        nearest_dists = nearest_dists.unsqueeze(-1)
        return nearest_vecs, nearest_dists
        
    def save_scene(self, filepath: str) -> None:
        """Save current scene state to JSON file
        
        Args:
            filepath: Path to save scene file
        """
        import json
        import numpy as np
        
        # Convert numpy arrays to strings for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert to list which is JSON serializable
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            return obj
            
        scene_data = {
            "objects": convert_arrays(self.objects),
        }
        
        with open(filepath, 'w') as f:
            json.dump(scene_data, f, indent=2)
            
    def load_scene(self, filepath: str) -> None:
        """Load scene state from JSON file
        
        Args:
            filepath: Path to scene file to load
        """
        import json
        import numpy as np
        
        with open(filepath, 'r') as f:
            scene_data = json.load(f)
            
        # Convert lists back to numpy arrays
        def restore_arrays(obj):
            if isinstance(obj, dict):
                if 'position' in obj:  # Convert position list to numpy array
                    obj['position'] = np.array(obj['position'])
                return {k: restore_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [restore_arrays(item) for item in obj]
            return obj
            
        scene_data = restore_arrays(scene_data)
            
        # Clear current scene
        self.objects.clear()
        self.meshes.clear()
        
        # Load object list
        self.objects = scene_data["objects"]
        
        # Rebuild meshes for all objects
        for obj in self.objects:
            # Create simple sphere mesh for each object
            # (Assuming objects have 'radius' property)
            self.create_box_mesh(size=obj.get("radius", 1.0))
        