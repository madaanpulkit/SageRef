# SageRef
SageRef: Single Image Reflection Removal


- [SageRef](#sageref)
- [Datasets Used](#datasets-used)
- [Evaluation](#evaluation)
  - [Data used for evaluation](#data-used-for-evaluation)
  - [Performance Metrics](#performance-metrics)
  - [Models used for comparisons](#models-used-for-comparisons)


# Datasets Used
- [SIR2 Benchmark Dataset](https://rose1.ntu.edu.sg/dataset/sir2Benchmark/)
- [CeilNet](https://github.com/fqnchina/CEILNet)
- [Dataset by Zhang et al](https://drive.google.com/drive/folders/1NYGL3wQ2pRkwfLMcV2zxXDV8JRSoVxwA?usp=sharing)

# Evaluation

## Data used for evaluation
#TODO

## Performance Metrics
- PSNR (Peak Signal-to-Noise Ratio): `torchmetrics.PeakSignalNoiseRatio`
- SSIM (Structural Similarity Index): `torchmetrics.StructuralSimilarityIndexMeasure` 
- LPIPS (Learned Perceptual Image Patch Similarity): `torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity`

## Models used for comparisons
- "Robust Reflection Removal with Reflection-free Flash-only Cues" (CVPR 2021) ([pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Lei_Robust_Reflection_Removal_With_Reflection-Free_Flash-Only_Cues_CVPR_2021_paper.pdf), [github](https://github.com/ChenyangLEI/flash-reflection-removal))
- "Trash or Treasure? An Interactive Dual-Stream Strategy for Single Image Reflection Separation" (NeurIPS 2021) ([pdf](https://proceedings.neurips.cc/paper/2021/file/cf1f78fe923afe05f7597da2be7a3da8-Paper.pdf), [github](https://github.com/mingcv/ytmt-strategy))
- "Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements" (CVPR 2019) ([pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Single_Image_Reflection_Removal_Exploiting_Misaligned_Training_Data_and_Network_CVPR_2019_paper.pdf),[github](https://github.com/Vandermode/ERRNet))
- "Single Image Reflection Removal with Perceptual Losses" (CVPR 2018) ([pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf), [github](https://github.com/ceciliavision/perceptual-reflection-removal))
