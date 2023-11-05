# Parasitic Egg Detection and Classification in Microscopic Images

This repository contains the code used in our project to address the [ICIP2020 Challenge Parasitic Egg Detection and Classification in Microscopic Images](https://icip2022challenge.piclab.ai/).

Our final model achieves the following performance metrics.

| Metric | Score |
|----------|----------|
| mIoU | 0.9377 |
| mF1 | 0.9689 |
| mAP@[IoU = 0.5:0.95] | 0.852 |
| mAP@[IoU = 0.5] | 0.936 |
| mAP@[IoU = 0.75] | 0.918 |

## Installation
The project was implemented using [MMDetection](https://github.com/open-mmlab/mmdetection) and [SAHI](https://github.com/obss/sahi) projects. 

To install MMDetection: (first go to the project working directory)
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

conda install pytorch torchvision -c pytorch

pip install -U openmim
mim install "mmengine>=0.7.0"
mim install "mmcv>=2.0.0rc4"
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .
```

To install SAHI: (go back to the project working directory)
```
git clone [https://github.com/open-mmlab/mmdetection.git](https://github.com/obss/sahi.git)
cd sahi
pip install -e .
```


## Data preparation
The script `data_preparation/n_fold_coco_slit_trainval.py` splits the available data into train and validation sets.
To cut images into patches, use script `data_preparation/n_fold_sahi_slice.py`. It creates a folder with sliced images (sliced) and an accompanying JSON file.

## Download the checkpoints
Download the original configuration file of RTMDet-tiny and the pretrained weights.
```
mkdir checkpoints
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest checkpoints
```

## Training
Adjust the file paths in `configs/basic_config.py` according to your data and annotation paths.
Use the `train.py` script available in MMDetection tools for training.
```
python mmdetection/tools/train.py configs/basic_config.py
```

## Inference
Use the `inference/sahi_predict_n_fold.py` script for running sliced inference on the test dataset after adjusting the paths. Set `novisual = False` if you want to save output images with bounding box annotations.
```
python inference/sahi_predict_n_fold.py
```

## Evaluation
Use the notebook `inference/test.ipynb` to compute metrics such as mIoU, mF1 and mAP.

## Results 



