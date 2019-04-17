# D2-Net: A Trainable CNN for Joint Detection and Description of Local Features

This repository contains the implementation of the following paper:

```text
"D2-Net: A Trainable CNN for Joint Detection and Description of Local Features".
M. Dusmanu, I. Rocco, T. Pajdla, M. Pollefeys, J. Sivic, A. Torii, and T. Sattler. CVPR 2019.
```

[Project page](https://dsmn.ml/publications/d2-net.html)
    
## Getting started

Python 3.6+ is recommended for running our code. [Conda](https://docs.conda.io/en/latest/) can be used to install the 
required packages:

```bash
conda install pytorch torchvision cudatoolkit=8.0 -c pytorch
conda install h5py imageio imagesize matplotlib numpy scipy tqdm
```

Due to a [bug](https://github.com/pytorch/pytorch/issues/15054) in a recent version of CUDNN, the runtime is severely 
affected on CUDA 9+. Thus, for the moment, we suggest sticking to CUDA 8 (or compiling PyTorch from sources with the 
latest CUDNN version).

## Downloading the models

The off-the-shelf Caffe VGG16 weights and their tuned counterpart can be downloaded by running:

```bash
mkdir models
wget https://dsmn.ml/files/d2-net/d2_ots.pth -O models/d2_ots.pth
wget https://dsmn.ml/files/d2-net/d2_tf.pth -O models/d2_tf.pth
```

## Feature extraction

`extract_features.py` can be used to extract D2 features for a given list of images. The singlescale features
require less than 6GB of VRAM for 1200x1600 images. The `--multiscale` flag can be used to extract multiscale features -
for this, we recommend at least 16GB of VRAM. 

The output format can be either [`npz`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) or `mat`. 
In either case, the feature files encapsulate two arrays: 

- `keypoints` - `N x 4` array containing the positions of keypoints `x, y`, the scales `s`, and the activation values 
`a`. The positions follow the COLMAP format, with the `X` axis pointing to the right and the `Y` axis to the bottom.
- `descriptors` - `N x 512` array containing the L2 normalized descriptors.

```bash
python extract_features.py --image_list_file images.txt (--multiscale)
```

## Tuning on MegaDepth

The training pipeline provided here is a PyTorch implementation of the TensorFlow code that was used to train the model 
available to download above. We ran a reproducibility study and the performance was similar - the TF weights yielded 
slightly better MMA (~3% difference) on the HPatches image pairs dataset. We might release new weights in the future!

### Downloading and preprocessing the MegaDepth dataset

After downloading the entire [MegaDepth](http://www.cs.cornell.edu/projects/megadepth/) dataset (including SfM models), 
`preprocess_megadepth.sh` can be used to retrieve the camera parameters and compute the overlap between images for all
scenes. 

```bash
cd megadepth_utils
bash preprocess_megadepth.sh /local/dataset/megadepth /local/dataset/megadepth/scenes_info
```

We are currently trying to find a way to release the processed scenes only in order to eliminate the need to download 
the SfM models. 

### Training

After downloading and preprocessing MegaDepth, the training can be started right away:

```bash
python train.py --use_validation --dataset_path /local/dataset/megadepth --scene_info_path /local/dataset/megadepth/scene_info
```

## BibTeX

If you use this code in your project, please cite the following paper:

```bibtex
@InProceedings{Dusmanu2019CVPR,
    author = {Dusmanu, Mihai and Rocco, Ignacio and Pajdla, Tomas and Pollefeys, Marc and Sivic, Josef and Torii, Akihiko and Sattler, Torsten},
    title = {{D2-Net: A Trainable CNN for Joint Detection and Description of Local Features}},
    booktitle = {Proceedings of the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2019},
}
```
