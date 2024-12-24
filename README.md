# Incremental-Learning-for-indoor-localization
## Overview  
This repository contains the implementation of the **ILCL algorithm** described in our paper: *"Intelligent Fingerprint-based Localization Scheme using CSI Images for Internet of Things"* and accepted by IEEE TNSE. ILCL leverages **incremental learning** and **channel state information (CSI)** to achieve efficient and accurate indoor localization without requiring full retraining when new fingerprint data are introduced.  

## Abstract  
Fingerprint-based indoor localization has become a crucial technology due to its wide availability, low hardware costs, and increasing demand for location-based services. However, existing methods face challenges such as:  
- **Low precision in positioning**.  
- **Time-consuming retraining** of models when the fingerprint database changes.  

ILCL addresses these challenges with an innovative localization scheme that combines incremental learning and CSI data processing.  

Key features of ILCL include:  
- **Offline Training Stage**:  
  - CSI phase data are extracted using a modified device driver and converted into CSI images.  
  - A convolutional neural network (CNN) processes CSI images to train the model weights.  
- **Online Localization Stage**:  
  - Incremental learning with a broad learning system (BLS) allows rapid adaptation to new input data without full model retraining.  
  - The estimated location is calculated using a probabilistic regression method based on BLS.  

Experimental results demonstrate that ILCL outperforms five state-of-the-art algorithms in two real-world indoor environments covering an area of over 200 m².  

## Usage  
The main scripts in this repository include:  
- `Main-lab.py`: Handles the offline training process, including CSI image generation and CNN model training for the lab scenario.  
- `Main-meeting.py`: This is for the meeting room scenario.

If you use this work for your research, you may want to cite
```
@article{zhu2022intelligent,
  title={{Intelligent Fingerprint-Based Localization Scheme Using CSI Images for Internet of Things}},
  author={Zhu, Xiaoqiang and Qu, Wenyu and Zhou, Xiaobo and Zhao, Laiping and Ning, Zhaolong and Qiu, Tie},
  journal={IEEE Transactions on Network Science and Engineering},
  volume={9},
  number={4},
  pages={2378--2391},
  year={2022}
}
```
