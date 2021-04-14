import argparse
import numpy as np
from PIL import Image
import torch
import math
from tqdm import tqdm

# Kapture is a pivot file format, based on text and binary files, used to describe SfM (Structure From Motion) and more generally sensor-acquired data
# it can be installed with
# pip install kapture
# for more information check out https://github.com/naver/kapture
import kapture
from kapture.io.records import get_image_fullpath
from kapture.io.csv import kapture_from_dir
from kapture.io.csv import get_csv_fullpath, keypoints_to_file, descriptors_to_file
from kapture.io.features import get_keypoints_fullpath, keypoints_check_dir, image_keypoints_to_file
from kapture.io.features import get_descriptors_fullpath, descriptors_check_dir, image_descriptors_to_file

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

# import imageio

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    '--kapture-root', type=str, required=True,
    help='path to kapture root directory'
)

parser.add_argument(
    '--preprocessing', type=str, default='caffe',
    help='image preprocessing (caffe or torch)'
)
parser.add_argument(
    '--model_file', type=str, default='models/d2_tf.pth',
    help='path to the full model'
)

parser.add_argument(
    '--max_edge', type=int, default=1600,
    help='maximum image size at network input'
)
parser.add_argument(
    '--max_sum_edges', type=int, default=2800,
    help='maximum sum of image sizes at network input'
)

parser.add_argument(
    '--multiscale', dest='multiscale', action='store_true',
    help='extract multiscale features'
)
parser.set_defaults(multiscale=False)

parser.add_argument(
    '--no-relu', dest='use_relu', action='store_false',
    help='remove ReLU after the dense feature extraction module'
)
parser.set_defaults(use_relu=True)

parser.add_argument("--max-keypoints", type=int, default=float("+inf"),
                                   help='max number of keypoints save to disk')

args = parser.parse_args()

print(args)

kdata = kapture_from_dir(args.kapture_root, 
skip_list= [kapture.GlobalFeatures,
            kapture.Matches,
            kapture.Points3d,
            kapture.Observations])

assert kdata.records_camera is not None
image_list = [filename for _, _, filename in kapture.flatten(kdata.records_camera)]
if kdata.keypoints is not None and kdata.descriptors is not None:
    image_list = [name for name in image_list if name not in kdata.keypoints or name not in kdata.descriptors]

if len(image_list) == 0:
    print('All features were already extracted')
    exit(0)
else:
    print(f'Extracting d2net features for {len(image_list)} images')

# Creating CNN model
model = D2Net(
    model_file=args.model_file,
    use_relu=args.use_relu,
    use_cuda=use_cuda
)

keypoints_dtype = None if kdata.keypoints is None else kdata.keypoints.dtype
descriptors_dtype = None if kdata.descriptors is None else kdata.descriptors.dtype

keypoints_dsize = None if kdata.keypoints is None else kdata.keypoints.dsize
descriptors_dsize = None if kdata.descriptors is None else kdata.descriptors.dsize

# Process the files
for image_name in tqdm(image_list, total=len(image_list)):
    img_path = get_image_fullpath(args.kapture_root, image_name)
    image = Image.open(img_path).convert('RGB')

    width, height = image.size

    resized_image = image
    resized_width = width
    resized_height = height

    max_edge = args.max_edge
    max_sum_edges = args.max_sum_edges
    if max(resized_width, resized_height) > max_edge:
        scale_multiplier = max_edge / max(resized_width, resized_height)
        resized_width = math.floor(resized_width * scale_multiplier)
        resized_height = math.floor(resized_height * scale_multiplier)
        resized_image = image.resize((resized_width, resized_height))
    if resized_width + resized_height > max_sum_edges:
        scale_multiplier = max_sum_edges / (resized_width + resized_height)
        resized_width = math.floor(resized_width * scale_multiplier)
        resized_height = math.floor(resized_height * scale_multiplier)
        resized_image = image.resize((resized_width, resized_height))

    fact_i = width / resized_width
    fact_j = height / resized_height

    resized_image = np.array(resized_image).astype('float')

    input_image = preprocess_image(
        resized_image,
        preprocessing=args.preprocessing
    )

    with torch.no_grad():
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales=[1]
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    if args.max_keypoints != float("+inf"):
        # keep the last (the highest) indexes
        idx_keep = scores.argsort()[-min(len(keypoints), args.max_keypoints):]
        keypoints = keypoints[idx_keep]
        descriptors = descriptors[idx_keep]

    
    if keypoints_dtype is None or descriptors_dtype is None:
        keypoints_dtype = keypoints.dtype
        descriptors_dtype = descriptors.dtype

        keypoints_dsize = keypoints.shape[1]
        descriptors_dsize = descriptors.shape[1]

        kdata.keypoints = kapture.Keypoints('d2net', keypoints_dtype, keypoints_dsize)
        kdata.descriptors = kapture.Descriptors('d2net', descriptors_dtype, descriptors_dsize)

        keypoints_config_absolute_path = get_csv_fullpath(kapture.Keypoints, args.kapture_root)
        descriptors_config_absolute_path = get_csv_fullpath(kapture.Descriptors, args.kapture_root)

        keypoints_to_file(keypoints_config_absolute_path, kdata.keypoints)
        descriptors_to_file(descriptors_config_absolute_path, kdata.descriptors)
    else:
        assert kdata.keypoints.type_name == 'd2net'
        assert kdata.descriptors.type_name == 'd2net'
        assert kdata.keypoints.dtype == keypoints.dtype
        assert kdata.descriptors.dtype == descriptors.dtype
        assert kdata.keypoints.dsize == keypoints.shape[1]
        assert kdata.descriptors.dsize == descriptors.shape[1]

    keypoints_fullpath = get_keypoints_fullpath(args.kapture_root, image_name)
    print(f"Saving {keypoints.shape[0]} keypoints to {keypoints_fullpath}")
    image_keypoints_to_file(keypoints_fullpath, keypoints)
    kdata.keypoints.add(image_name)

    
    descriptors_fullpath = get_descriptors_fullpath(args.kapture_root, image_name)
    print(f"Saving {descriptors.shape[0]} descriptors to {descriptors_fullpath}")
    image_descriptors_to_file(descriptors_fullpath, descriptors)
    kdata.descriptors.add(image_name)

if not keypoints_check_dir(kdata.keypoints, args.kapture_root) or \
        not descriptors_check_dir(kdata.descriptors, args.kapture_root):
    print('local feature extraction ended successfully but not all files were saved')
