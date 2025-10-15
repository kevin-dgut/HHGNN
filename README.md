# Multiscale Altases Based Hierarchical Heterogeneous Graph Learning for Brain Disorder Diagnosis
[2025/07] The code will be publicly available once the paper is accepted. Thanks for your attention!
 This is the official PyTorch implementation of HHGNN from the paper "Multi-scale Atlases Based Hierarchical Heterogeneous Graph Learning for Brain Disorder Diagnosis".
 
## Overview
In this paper, we introduce a novel framework for brain disorder diagnosis based on the hierarchical and heterogeneous BFC graphs extracted from multi-scale atlases. Based on the extensive experiments on fMRI data of 2,132 subjects from ADNI, ABIDE, ADHD-200 and STAR datasets, we have demonstrated the superior performance of our approach for various brain disorder diagnosis tasks, especially for clinical promising early MCI diagnosis task. The experimental results have highlighted the necessity of incorporating hierarchical and heterogeneous BFC graph modeling by identifying the intra-scale, inter-scale overlapped and inter-scale non-overlapped edges separately. One of the important ﬁndings of this work is that functional connections of ROIs in diﬀerent atlas scales should not be treated as the same homogeneous structure for brain disorder diagnosis.

## Instructions
The public datasets [ADNI](https://adni.loni.usc.edu/), [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/), and [ADHD-200](http://fcon_1000.projects.nitrc.org/indi/adhd200/) used in the paper are downloaded from their official websites.


## Requirements
- Python
- PyTorch
- DGL
- Numpy
- Scipy
- OS
- etc.

## Optional
- GPU support is recommended for faster training
