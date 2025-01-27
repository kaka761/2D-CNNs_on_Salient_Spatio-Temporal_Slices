# 2D-CNNs_on_Salient_Spatio-Temporal_Slices

This repo is for:

**Paper 1:** ['Video Understanding Using 2D-CNNs on Salient Spatio-Temporal Slices'](https://link.springer.com/chapter/10.1007/978-3-031-72338-4_18) by Yaxin Hu & Erhardt Barth


## Introduction

Video understanding remains a challenge even with advanced deep-learning methods that typically sample a few frames from which spatial and temporal features are extracted. Such down-sampling often leads to the loss of critical temporal information. Moreover, current state-of-the-art methods involve high computational costs. 2D Convolutional Neural Networks (2D-CNNs) have proven to be effective at capturing spatial features of images, but cannot make use of temporal information. To address these challenges, we propose to use 2D-CNNs not only on images, i.e. xy-slices of the video, but on salient spatio-temporal xt and yt slices to efficiently capture both spatial and temporal information of the entire video. As 2D-CNNs are known to extract local spatial orientation in xy, they can now extract motion, which is a local orientation in xt and yt. We complement the approach with a simple strategy for sampling the most informative slices and show that we can outperform alternative approaches in a number of tasks, especially in cases in which the actions are defined by their dynamics, i.e., by spatio-temporal patterns.

### Methodology

**Slicing Videos**

<div align=left>
<img src="https://github.com/kaka761/2D-CNNs_on_Salient_Spatio-Temporal_Slices/blob/master/xyt.png" align="center" width=55% />
</div>

**Architecture**

<div align=left>
<img src="https://github.com/kaka761/2D-CNNs_on_Salient_Spatio-Temporal_Slices/blob/master/Arch.png" align="center" width=55% />
</div>

**Salient & Non-Salient Slices**

<div align=left>
<img src="https://github.com/kaka761/2D-CNNs_on_Salient_Spatio-Temporal_Slices/blob/master/salient.png" align="center" width=55% />
</div>

**Examples of Action Recognition Tasks**

<div align=left>
<img src="https://github.com/kaka761/2D-CNNs_on_Salient_Spatio-Temporal_Slices/blob/master/AR.png" align="center" width=55% />
</div>

**Examples of Hand Geature Recognition Tasks**

<div align=left>
<img src="https://github.com/kaka761/2D-CNNs_on_Salient_Spatio-Temporal_Slices/blob/master/HGR.png" align="center" width=55% />
</div>

## Cite
<details>
<summary>BibTeX entry for citation.</summary>
<pre>
@inproceedings{hu2024video,
  title={Video Understanding Using 2D-CNNs on Salient Spatio-Temporal Slices},
  author={Hu, Yaxin and Barth, Erhardt},
  booktitle={International Conference on Artificial Neural Networks},
  pages={256--270},
  year={2024},
  organization={Springer}
}
</pre>
</details>
