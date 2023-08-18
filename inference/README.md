# Inference detail for our LPCV 2023 solution


## Inference Environment
### Hardware
- NVIDIA Jetson Nano 2GB

### Software
- [23LPCVC_Segmentation_Track-Sample_Solution](https://github.com/lpcvai/23LPCVC_Segmentation_Track-Sample_Solution)

## Convert
### Step 1: Graph optimization
Use [onnx-modifier](https://github.com/ZhangGe6/onnx-modifier) to remove the redundant operators in the onnx model, such as `softmax`, `argmax`.

### Step 2: Convert to TensorRT Model
Transfer the modified onnx to Jeston Nano.
Convert to TensorRT on Jeston Nano by following command:
```
/usr/src/tensorrt/bin/trtexec --onnx=<xxxx.onnx> --saveEngine=<xxxx.trt> --best
```

## Post-process
Goto [process.py](process.py) for post-process detail.
