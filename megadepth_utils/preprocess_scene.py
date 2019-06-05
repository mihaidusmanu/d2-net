import argparse

import imagesize

import numpy as np

import os

parser = argparse.ArgumentParser(description='MegaDepth preprocessing script')

parser.add_argument(
    '--base_path', type=str, required=True,
    help='path to MegaDepth'
)
parser.add_argument(
    '--scene_id', type=str, required=True,
    help='scene ID'
)
parser.add_argument(
    '--subscene_id', type=str, required=True,
    help='subscene ID'
)

parser.add_argument(
    '--output_path', type=str, required=True,
    help='path to the output directory'
)

args = parser.parse_args()

base_path = args.base_path
scene_id = args.scene_id
subscene_id = args.subscene_id

base_sfm_path = os.path.join(
    base_path, 'MegaDepth_v1_SfM'
)
base_depth_path = os.path.join(
    base_path, 'phoenix/S6/zl548/MegaDepth_v1'
)

sparse_path = os.path.join(
    base_sfm_path, scene_id, 'sparse/manhattan', subscene_id
)
if not os.path.exists(sparse_path):
    exit()
depths_path = os.path.join(
    base_depth_path, scene_id, 'dense' + subscene_id, 'depths'
)
if not os.path.exists(depths_path):
    exit()
raw_images_path = os.path.join(
    base_sfm_path, scene_id, 'images'
)
if not os.path.exists(raw_images_path):
    exit()
images_path = os.path.join(
    base_depth_path, scene_id, 'dense' + subscene_id, 'imgs'
)
if not os.path.exists(images_path):
    exit()

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

# Process cameras.txt
with open(os.path.join(sparse_path, 'cameras.txt'), 'r') as f:
    raw = f.readlines()[3 :]  # skip the header

camera_intrinsics = {}
for camera in raw:
    camera = camera.split(' ')
    camera_intrinsics[int(camera[0])] = [float(elem) for elem in camera[2 :]]

# Process images.txt
with open(os.path.join(sparse_path, 'images.txt'), 'r') as f:
    raw = f.readlines()[4 :]  # skip the header

image_id_to_idx = {}
image_names = []
raw_pose = []
camera = []
points3D_id_to_2D = []
for idx, (image, points) in enumerate(zip(raw[:: 2], raw[1 :: 2])):
    image = image.split(' ')
    points = points.split(' ')

    image_id_to_idx[int(image[0])] = idx

    image_name = image[-1].strip('\n')
    image_names.append(image_name)

    raw_pose.append([float(elem) for elem in image[1 : -2]])
    camera.append(int(image[-2]))
    current_points3D_id_to_2D = {}
    for x, y, point3D_id in zip(points[:: 3], points[1 :: 3], points[2 :: 3]):
        if int(point3D_id) == -1:
            continue
        current_points3D_id_to_2D[int(point3D_id)] = [float(x), float(y)]
    points3D_id_to_2D.append(current_points3D_id_to_2D)
n_images = len(image_names)

# Image and depthmaps paths
image_paths = []
depth_paths = []
fact_x = []
fact_y = []
for image_name in image_names:
    # Path to the undistorted image
    image_path = os.path.join(images_path, image_name)
    raw_image_path = os.path.join(raw_images_path, image_name)
    if os.path.exists(image_path):
        image_paths.append(raw_image_path[len(base_path) + 1 :])
    else:
        image_paths.append(None)

    # Path to the depth file
    depth_path = os.path.join(
        depths_path, '%s.h5' % os.path.splitext(image_name)[0]
    )
    if os.path.exists(depth_path):
        # Check if depth map or background / foreground mask
        file_size = os.stat(depth_path).st_size
        # Rough estimate - 75KB might work as well
        if file_size < 100 * 1024:
            depth_paths.append(None)
        else:
            depth_paths.append(depth_path[len(base_path) + 1 :])
    else:
        depth_paths.append(None)

    if image_paths[-1] is None:
        fact_x.append(None)
        fact_y.append(None)
        continue

    raw_size = imagesize.get(raw_image_path)
    size = imagesize.get(image_path)
    fact_x.append(size[0] / raw_size[0])
    fact_y.append(size[1] / raw_size[1])

# Camera configuration
intrinsics = []
poses = []
principal_axis = []
for idx, image_name in enumerate(image_names):
    if image_paths[idx] is None:
        intrinsics.append(None)
        poses.append(None)
        principal_axis.append([0, 0, 0])
        continue
    image_intrinsics = camera_intrinsics[camera[idx]]
    K = np.zeros([3, 3])
    K[0, 0] = image_intrinsics[2] * fact_x[idx]
    K[0, 2] = image_intrinsics[3] * fact_x[idx]
    K[1, 1] = image_intrinsics[2] * fact_y[idx]
    K[1, 2] = image_intrinsics[4] * fact_y[idx]
    K[2, 2] = 1
    intrinsics.append(K)

    image_pose = raw_pose[idx]
    qvec = image_pose[: 4]
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    R = np.array([
        [
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * z * w,
            2 * x * z + 2 * y * w
        ],
        [
            2 * x * y + 2 * z * w,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * x * w
        ],
        [
            2 * x * z - 2 * y * w,
            2 * y * z + 2 * x * w,
            1 - 2 * x * x - 2 * y * y
        ]
    ])
    principal_axis.append(R[2, :])
    t = image_pose[4 : 7]
    # World-to-Camera pose
    current_pose = np.zeros([4, 4])
    current_pose[: 3, : 3] = R
    current_pose[: 3, 3] = t
    current_pose[3, 3] = 1
    # Camera-to-World pose
    # pose = np.zeros([4, 4])
    # pose[: 3, : 3] = np.transpose(R)
    # pose[: 3, 3] = -np.matmul(np.transpose(R), t)
    # pose[3, 3] = 1
    poses.append(current_pose.flatten())
principal_axis = np.array(principal_axis)
angles = np.rad2deg(np.arccos(
    np.clip(
        np.dot(principal_axis, np.transpose(principal_axis)),
        -1, 1
    )
))

# Compute overlap score
overlap_matrix = np.full([n_images, n_images], -1.)
scale_ratio_matrix = np.full([n_images, n_images], -1.)
for idx1 in range(n_images):
    if image_paths[idx1] is None or depth_paths[idx1] is None:
        continue
    for idx2 in range(idx1 + 1, n_images):
        if image_paths[idx2] is None or depth_paths[idx2] is None:
            continue
        matches = (
            points3D_id_to_2D[idx1].keys() &
            points3D_id_to_2D[idx2].keys()
        )
        min_num_points3D = min(
            len(points3D_id_to_2D[idx1]), len(points3D_id_to_2D[idx2])
        )
        overlap_matrix[idx1, idx2] = len(matches) / min_num_points3D
        overlap_matrix[idx2, idx1] = len(matches) / min_num_points3D
        if len(matches) == 0:
            continue
        pos1 = np.array([
            points3D_id_to_2D[idx1][point3D_id] for point3D_id in matches
        ])
        pos2 = np.array([
            points3D_id_to_2D[idx2][point3D_id] for point3D_id in matches
        ])
        scale1 = (
            (np.max(pos1[:, 0]) - np.min(pos1[:, 0])) *
            (np.max(pos1[:, 1]) - np.min(pos1[:, 1]))
        )
        scale2 = (
            (np.max(pos2[:, 0]) - np.min(pos2[:, 0])) *
            (np.max(pos2[:, 1]) - np.min(pos2[:, 1]))
        )
        if scale2 == 0 or scale1 == 0:
            continue
        scale_ratio_matrix[idx1, idx2] = np.sqrt(scale1 / scale2)
        scale_ratio_matrix[idx2, idx1] = np.sqrt(scale2 / scale1)

np.savez(
    os.path.join(args.output_path, '%s.%s.npz' % (scene_id, subscene_id)),
    image_paths=image_paths,
    depth_paths=depth_paths,
    points3D_id_to_2D=points3D_id_to_2D,
    intrinsics=intrinsics,
    poses=poses,
    fact_x=fact_x,
    fact_y=fact_y,
    overlap_matrix=overlap_matrix,
    scale_ratio_matrix=scale_ratio_matrix,
    angles=angles
)
