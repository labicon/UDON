import argparse
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

from config import load_config
from datasets.dataset import get_dataset

def run_tsdf_fusion(cfg, save_path):
    dataset = get_dataset(cfg)
    H, W = dataset.H, dataset.W
    fx, fy, cx, cy = dataset.fx, dataset.fy, dataset.cx, dataset.cy
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    voxel_length = 0.03
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=0.12,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i, frame in enumerate(tqdm(dataset)):
        rgb, depth, c2w = frame["rgb"].cpu().numpy(), frame["depth"].cpu().numpy(), frame["c2w"].cpu().numpy()
        rgb = rgb * 255
        rgb = rgb.astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        depth = depth.astype(np.float32)
        depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1.0,
                                                                  depth_trunc=8.0,
                                                                  convert_rgb_to_intensity=False)
        # to OpenCV
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        # requires w2c
        w2c = np.linalg.inv(c2w)
        volume.integrate(rgbd, K, w2c)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print("Saving mesh...")
    o3d.io.write_triangle_mesh(save_path, mesh)
    return mesh

def create_camera_poses(mesh_path, save_dir, num_random_views=0):
    """
    create camera poses for a given mesh, use completely standard views and some random views.
    Args:
        mesh_path (str): Path to the mesh file.
        save_dir (str): Directory to save the camera poses.
        num_random_views (int): Number of random views to generate.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # load mesh and compute its bounding box
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    
    # set the camera radius based on the bounding box
    radius = max(extent) * 1.5  
    
    #  Create standard views
    standard_views = [
        ("front", 0, np.pi/2),             # Front view
        ("back", np.pi, np.pi/2),          # Back view
        ("left", np.pi/2, np.pi/2),        # Left view
        ("right", -np.pi/2, np.pi/2),      # Right view
        ("top", 0, 0.1),                   # Top view (slightly tilted)
        ("front_right", np.pi*7/4, np.pi/2), # Front-right view
        ("front_left", np.pi*1/4, np.pi/2),  # Front-left view
        ("back_right", np.pi*5/4, np.pi/2),  # Back-right view
        ("back_left", np.pi*3/4, np.pi/2),   # Back-left view
        ("diagonal1", np.pi*1/4, np.pi/4),   # Upper front-right view
        ("diagonal2", np.pi*3/4, np.pi/4),   # Upper front-left view
        ("diagonal3", np.pi*5/4, np.pi/4),   # Upper back-left view
        ("diagonal4", np.pi*7/4, np.pi/4),   # Upper back-right view
    ]
    
    print(f"Generating {len(standard_views)} standard views for the scene...")
    
    # Generate camera poses for standard views
    for name, theta, phi in standard_views:
        # Calculate camera position
        eye = center + np.array([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi)
        ])
        
        
        # Camera looks at the center point
        forward = center - eye
        forward = forward / np.linalg.norm(forward)
        
        # Define world-space up direction
        world_up = np.array([0.0, 0.0, 1.0])
        
        # Calculate camera coordinate system
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Build camera-to-world transformation matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = -up  # Note: camera's y-axis typically points down
        c2w[:3, 2] = forward
        c2w[:3, 3] = eye
        
        # Save camera pose
        np.savetxt(os.path.join(save_dir, f"{name}.txt"), c2w)
        print(f"Saved {name} view")
    
    
    # If random views are requested
    if num_random_views > 0:
        print(f"Generating {num_random_views} random views...")
        for i in range(num_random_views):
            # Generate random angles
            theta = np.random.uniform(0, 2 * np.pi)  # Horizontal angle
            phi = np.random.uniform(np.pi/6, np.pi/2.5)  # Vertical angle, avoid too low or too high
            
            # Calculate camera position
            eye = center + np.array([
                radius * np.sin(phi) * np.cos(theta),
                radius * np.sin(phi) * np.sin(theta),
                radius * np.cos(phi)
            ])
            
            # Camera looks at the center point
            forward = center - eye
            forward = forward / np.linalg.norm(forward)
            
            # Define world-space up direction
            world_up = np.array([0.0, 0.0, 1.0])
            
            # Calculate camera coordinate system
            right = np.cross(forward, world_up)
            right = right / np.linalg.norm(right)
            
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            # Build camera-to-world transformation matrix
            c2w = np.eye(4)
            c2w[:3, 0] = right
            c2w[:3, 1] = -up  # Note: camera's y-axis typically points down
            c2w[:3, 2] = forward
            c2w[:3, 3] = eye
            
            # Save camera pose
            np.savetxt(os.path.join(save_dir, f"random_{i:03d}.txt"), c2w)
            
        print(f"Generated {num_random_views} random views")
    
    print(f"All camera poses saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file.")
    parser.add_argument("--data_dir", type=str,
                        help="Path to dataset sequence. This has higher priority.")
    parser.add_argument("--random_views", type=int, default=0,
                        help="Number of additional random views to generate.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir is not None:
        cfg["data"]["datadir"] = args.data_dir
    datadir = cfg["data"]["datadir"]

    tsdf_fusion_file = os.path.join(datadir, "tsdf_fusion.ply")
    if not os.path.exists(tsdf_fusion_file):
        print("TSDF-Fusion mesh not created, creating now...")
        run_tsdf_fusion(cfg, tsdf_fusion_file)
    else:
        print(f"Using existing TSDF mesh: {tsdf_fusion_file}")

    save_dir = os.path.join(datadir, "virtual_cameras")
    create_camera_poses(tsdf_fusion_file, save_dir, num_random_views=args.random_views)