# Solution for 2023 LOW-POWER COMPUTER VISION CHALLENGE

The solution of 2nd place (most accurate, speed within the top five) of [Segmentation track in 2023 Low-Power Computer Vision Challenge (LPCVC)](https://lpcv.ai/2023LPCVC/introduction).

## Description
The model submitted for the 2023 LPCVC and implementation code for training and exportation.
* Task: Semantic Segmentation
* Algorithm: TopFormer

## Methodology
- Our model selection is based on Topformer and makes some modifications upon it.

    For this competition, we modified Topformer_tiny as follows:
    1. Reduce the input resolution to 288; 
    2. The number of dynamic pyramid layers in the model is reduced from 9 to 8; 
    3. The number of transformer blocks is reduced from 4 to 3.

- Then, We use Topformer_base as the teacher model to perform knowledge distillation on the modified Topformer_tiny.

## Model

|Model|Input Image Size|Accuracy|Latency|Description|Download|
| - | - | - | - | - | -|
|Topformer_base|512x512|61.1%|70ms|teacher model|[Download Link](https://drive.google.com/file/d/1k_7BsVcLyLxHr8f8iFoxQHlUiEHtgEAE/view?usp=drive_link)|
|Topformer_tiny_modifed|288x288|15.3ms|54.5%|undistilled student model|[Download Link](https://drive.google.com/file/d/1ks1EMBykaXH7-n772FmlJ-XMFVtgaBl_/view?usp=drive_link)|
|Topformer_tiny_modifed|288x288|15.3ms|55.4%|distilled student model|[Download Link](https://drive.google.com/file/d/1ZZzZCwENig7wBh9W82HwbZ8jcZ6UNj7D/view?usp=drive_link)|
|Topformer_tiny_encoder|-|-|-|backbone|[Download Link](https://drive.google.com/file/d/1Y3TswTPNFIIqHUVae8A30-rLW1IWR1zb/view?usp=drive_link)|

## How to install and setup
### Requirements

- python 3.8.8
- pytorch 1.5+
- mmcv-full==1.4.0
- mmsegmentation==0.19.0
- mmrazor==0.3.1
- mmcls==0.19.0
- mmdeploy==0.14.0

### Installation

```
pip install -U openmim
mim install 'mmengine==0.7.3'
mim install "mmcv-full==1.4.0"
pip install mmsegmentation==0.19.0
pip install mmrazor==0.3.1
pip install mmcls==0.19.0
pip install mmdeploy==0.14.0
```
in the path of the source program
```
pip install -v -e .
```

## How to run
### Step 0: Dataset path
Setup the dataset path in 'local_configs/_base_/datasets/LPCVC.py'->'data_root'.
Setup the dataset path in 'local_configs/_base_/datasets/LPCVC_distill.py'->'data_root'.

### Step 1: Train
Train your model by by following command:
```
python tools_mmseg/train.py local_configs/ttopformer/topformer_tiny_288x288_160k_2x8_ade20k.py --work-dir <path-to-save-checkpoints>
```
<!-- Such as:
```
python tools_mmseg/train.py local_configs/topformer/topformer_tiny_288x288_160k_2x8_ade20k.py --work-dir output/test
``` -->

### Step 2: Select the best model as the student model

Select the model with the highest accuracy from all checkpoints and use it as the student model by following command:

```
python tools_mmseg/mytest.py local_configs/ttopformer/topformer_tiny_288x288_160k_2x8_ade20k.py --checkpoint <checkpoint-path> --eval mDice
```
<!-- Such as:
```
python tools_mmseg/mytest.py local_configs/ttopformer/topformer_tiny_288x288_160k_2x8_ade20k.py --checkpoint output/tiny_288_8tp_3trans_7-31/iter_62500.pth --eval mDice
``` -->

### Step 3: Knowledge distillation

Modify the distillation configuration file: 'local_configs/distill/cwd_seg_topformer_512b_distill_288t.py'.
Then, run the following command:
```
python tools_mmraz/mmseg/train_mmseg.py local_configs/distill/cwd_seg_topformer_512b_distill_288t.py --work-dir <path-to-save-checkpoints>
```
<!-- Such as:
```
python tools_mmraz/mmseg/train_mmseg.py local_configs/distill/cwd_seg_topformer_512b_distill_288t.py --work-dir distill_output/test
``` -->

Select the model with the highest accuracy from all checkpoints by following command:
```
python tools_mmraz/mmseg/mytest_mmseg.py local_configs/distill/cwd_seg_topformer_512b_distill_288t.py --checkpoint <checkpoint-path> --eval mDice
```
<!-- Such as:
```
python tools_mmraz/mmseg/mytest_mmseg.py local_configs/distill/cwd_seg_topformer_512b_distill_288t.py --checkpoint distill_output/512b_288t_tau2/iter_35000.pth --eval mDice
``` -->

### Step 4: Convert to ONNX
Modify the path of the best model into the python file: 'split_mmrazor_pth.py'. 
Then, run 'split_mmrazor_pth.py', You can get a new 'pth' file. 
Convert this 'pth' file to onnx file by following command:
```
python tools_mmdep/deploy.py local_configs/deploy/segmentation_onnxruntime_static-288x288.py local_configs/topformer/topformer_tiny_288x288_160k_2x8_ade20k.py <checkpoint-path> <dummy-data-path> --work-dir <path-to-save-onnx>
```
<!-- Such as:
```
python tools_mmdep/deploy.py local_configs/deploy/segmentation_onnxruntime_static-288x288.py local_configs/topformer/topformer_tiny_288x288_160k_2x8_ade20k.py tiny_288_8tp_3trans_dist.pth local_configs/train_0373.png --work-dir onnx_model
``` -->

### Step 5: Graph optimization and convert to TensorRT
Use 'onnx-modifier' to remove the redundant operators in the onnx model, such as softmax, argmax.
(You can get 'onnx-modifier.exe' from https://github.com/ZhangGe6/onnx-modifier)

Then, transfer the modified onnx to jeston nano and convert to TensorRT by following command:
```
/usr/src/tensorrt/bin/trtexec --onnx=<onnx-path> --saveEngine=<path-to-save-trt> --best
```


## License
Apache License 2.0 