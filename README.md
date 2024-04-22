
### Official reproduction of GVTNet

### Test images 

- Download the pre-trained [models](https://pan.baidu.com/s/1L4tQsvgeiVJhuZ3nMiX8TA)(3mxu) and place them in `experiments/pretrained_models/`.

  We provide pre-trained models for image SR: X4,X8

- Put your dataset ( LR images) in `datasets/`. Some test images are in this folder.

- Run the following scripts. The testing  is in `inference/inference_GVTNet.py (e.g., [inference_GVTNet.py](inference/inference_GVTNet.py)).
### visual comparison
<p align="center">
  <img src="https://github.com/continueyang/GVTNet/blob/main/assets/output_page_0.png?raw=true" width="480">
</p>

## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR).