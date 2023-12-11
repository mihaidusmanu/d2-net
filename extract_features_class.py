import numpy as np

import imageio

import torch

from tqdm import tqdm

from PIL import Image

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale



class ExtractFeatures:
    def __init__(self):
        self.init_params()
        self.create_model()

    def init_params(self):
        # CUDA
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        class DefaultArgs:
            def __init__(self):
                   self.preprocessing="torch" 
                   self.model_file="models/d2_tf.pth" 
                   self.max_edge=1600
                   self.max_sum_edges=2800
                   self.multiscale=False
                   self.use_relu=True

        self.args = DefaultArgs()
        print("Args", self.args)

    def create_model(self):
        # Creating CNN model
        self.model = D2Net(
            model_file=self.args.model_file,
            use_relu=self.args.use_relu,
            use_cuda=self.use_cuda
        )

    # image_list_file: path to a file containing a list of images to process
    def process(self, image_list_file: str):
        # Process the file
        ret_list = []
        args = self.args
        with open(image_list_file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, total=len(lines)):
                path = line.strip()

                image = imageio.v2.imread(path)
                if len(image.shape) == 2:
                    image = image[:, :, np.newaxis]
                    image = np.repeat(image, 3, -1)

                resized_image = image
                if max(resized_image.shape) > args.max_edge:
                    resized_image = Image.fromarray(resized_image).resize(args.max_edge / max(resized_image.shape)).astype('float')
                if sum(resized_image.shape[: 2]) > args.max_sum_edges:
                    resized_image = Image.fromarray(resized_image).resize(args.max_sum_edges / sum(resized_image.shape[: 2])).astype('float')
                        
     
                fact_i = image.shape[0] / resized_image.shape[0]
                fact_j = image.shape[1] / resized_image.shape[1]

                input_image = preprocess_image(
                    resized_image,
                    preprocessing=args.preprocessing
                )
                with torch.no_grad():
                    if args.multiscale:
                        keypoints, scores, descriptors = process_multiscale(
                            torch.tensor(
                                input_image[np.newaxis, :, :, :].astype(np.float32),
                                device=self.device
                            ),
                            self.model
                        )
                    else:
                        keypoints, scores, descriptors = process_multiscale(
                            torch.tensor(
                                input_image[np.newaxis, :, :, :].astype(np.float32),
                                device=self.device
                            ),
                            self.model,
                            scales=[1]
                        )

                # Input image coordinates
                keypoints[:, 0] *= fact_i
                keypoints[:, 1] *= fact_j
                # i, j -> u, v
                keypoints = keypoints[:, [1, 0, 2]]

                ret_list.append({"keypoints": keypoints, "scores": scores, "descriptors": descriptors})
            return ret_list
        
