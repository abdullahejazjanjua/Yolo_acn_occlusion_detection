# Introduction

This repository contains my implementation of **YOLO-ACN**, as described in the paper [*YOLO-ACN: Focusing on Small Target and Occluded Object Detection*](https://ieeexplore.ieee.org/document/9303478). YOLO-ACN is a novel method for enhancing the detection of small and occluded objects, an important challenge in real-world object detection tasks. 

## Current Status

**Work in Progress**  
This repository is under active development. The following components have been implemented:

- **Backbone**
- **Neck**
- **CBAM (Convolutional Block Attention Module)**
- **Head**
- **YOLO-ACN**

## Key Features

- **Focus on Small Objects**: YOLO-ACN improves detection accuracy for small and occluded objects, which are often challenging for traditional detection models.
- **Adaptive Contextual Networks**: The use of contextual information to better handle occlusions and small object detections.
- **State-of-the-Art Performance**: Enhances the performance of standard YOLO models by leveraging advanced techniques.


## Model Components

- **Backbone**: Extracts high-level feature maps from input images.
- **Neck**: Enhances the feature maps by improving receptive fields.
- **CBAM (Convolutional Block Attention Module)**: Attention mechanism to refine feature maps.

## Results

| Model         | mAP (Mean Average Precision) |
|---------------|-------------------------------|
| YOLOv8n        | 37%                            |
| YOLO-ACN      | Y%                            |


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The YOLO-ACN paper: [YOLO-ACN: Focusing on Small Target and Occluded Object Detection](https://ieeexplore.ieee.org/document/9303478)
- YOLOv8 and other object detection models
