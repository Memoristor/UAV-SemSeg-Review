# UAV-SemSeg-Review
Methods and datasets on semantic segmentation for Unmanned Aerial Vehicle remote sensing images: A review

## Briefly

* The repository contains the official code for the journal paper "Methods and Datasets on Semantic Segmentation for Unmanned Aerial Vehicle Remote Sensing Images: A Review", which compares the segmentation performance and inference efficiency of current popular semantic segmentation methods for UAV images.
* We conducted comparison experiments on two high-resolution UAV optical RGB image datasets: (a) UAVid, and (b) FloodNet. All images will be resized to $512\times512$ during the training, validation and testing phases since the computation loads for large-scale images may exceed the resource allocation limit.

## Environment

### Runtime Environment

* Ubuntu 18.04.5 LTS (8G RAM)
* PyTorch 1.10.0
* CUDA 11.1+
* NVIDIA RTX3090

### Conda Environment

```shell
# Create your conda environment
conda create -n pytorch python=3.9

# Activate conda environment and upgrade pip
conda/source activate pytorch
python -m pip install --upgrade pip

# Install packages and dependencies
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install imgaug matplotlib tqdm prettytable tensorboardX 

# Install pydensecrf
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## How to use

### Step1: Prepare the pre-trained model

The pre-trained models required for HRNet can be found in the [HRNet-Semantic-Segmentation
](https://github.com/HRNet/HRNet-Semantic-Segmentation) repository. The downloaded pre-trained models should be placed in the `models/pretrained` folder in the following order.

```
models/
├── __init__.py
├── bisenet_v1.py
├── ...
├── pretrained
│   ├── hrnetv2_w48_imagenet_pretrained.pth
│   ├── R50+ViT-B_16.npz
│   ├── R50+ViT-L_32.npz
│   └── ViT-B_16.npz
├── pspnet_resnet.py
├── ...
```

### Step2: Train/Valid Phase

It is recommended to start training or testing by scripts. Let's take the training phase as an example,

* Set the hyper-parameters for training phase, such as learning rate and weight decay

* Specify the models you need to train, e.g.
  
  ```shell
  python ../main.py \
    ...
    --model "SegNetVGG16" "FCN8s" "FCN16s" "PSPNetVGG16" "PSPNetResNet50" \
    ...
  ```

* Specify the running GPU and run the script in terminal 
  
  ```shell
  ./uavid_train_valid.sh 0,1
  ```

### Step3: Test Phase

It is also allowed to set whether to enable `DenseCRF` or `ConvCRF` before running the tests.

  * Specify whether to enable `DenseCRF` (--densecrf) or `ConvCRF` (--convcrf), e.g.

    ```
    python ../main.py \
      ...
      --model "SegNetVGG16" "FCN8s" "FCN16s" "PSPNetVGG16" "PSPNetResNet50" \
      --densecrf
      ...
    ```

  * Specify the running GPU and run the script in terminal 

    ```
    ./uavid_test_only.sh 0
    ```

## Citation

  ```
  @article{cheng_methods_2024,
    title = {Methods and datasets on semantic segmentation for Unmanned Aerial Vehicle remote sensing images: A review},
    author = {Cheng, Jian and Deng, Changjian and Su, Yanzhou and An, Zeyu and Wang, Qi},
    year = {2024},
    month = may,
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {211},
    pages = {1--34},
    issn = {0924-2716}
  }
  ```
