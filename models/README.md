## Place pretrained models here. 

We provide two pretrained models:

1. `RRDB_ESRGAN_x4.pth`: the final ESRGAN model we used in our [paper](https://arxiv.org/abs/1809.00219). 
2. `RRDB_PSNR_x4.pth`: the PSNR-oriented model with **high PSNR performance**.

*Note that* the pretrained models are trained under the `MATLAB bicubic` kernel. 
If the downsampled kernel is different from that, the results may have artifacts.
# Evaluation metrics 
1. Peak Signal-to-Noise Ratio (PSNR)

Measures the quality of the enhanced image compared to the original.
Higher PSNR indicates better image reconstruction with less distortion.
Unit: dB (Decibels).

2. Structural Similarity Index Measure (SSIM)

Evaluates the similarity between the original and enhanced images.
Considers brightness, contrast, and structure.
Value ranges from 0 to 1.
A value closer to 1 indicates better image quality.
