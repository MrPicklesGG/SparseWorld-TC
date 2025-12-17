<p align="center">
  <h2 align="center">SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model</h2>
  <p align="center">
    <a >Jiayuan Du*</a><sup>1, </sup><sup>2</sup>
    &nbsp;·&nbsp;
    <a >Yiming Zhao*</a><sup>2</sup>
    &nbsp;·&nbsp;
    <a >Zhenglong Guo</a><sup>2</sup>
    &nbsp;·&nbsp;
    <a >Yong Pan</a><sup>2</sup>
    &nbsp;·&nbsp;
    <a >Wenbo Hou</a><sup>2</sup>
    &nbsp;·&nbsp;
    <a >Zhihui Hao</a><sup>2</sup>
    &nbsp;·&nbsp; <br>
    <a >Kun Zhan</a><sup>2</sup>
    &nbsp;·&nbsp;
    <a >Qijun Chen#</a><sup>1</sup>
  </p>
  <p align="center">
    <sup>1</sup>Tongji University
    <br>
    <sup>2</sup>Li Auto Inc.
    <br>
    (* Equal contribition, # Corresponding author)
  </p>
  <!-- <h3 align="center"> 2026</h3> -->
  <h3 align="center"><a href="https://arxiv.org/abs/2511.22039">Paper</a> | <a href="https://drive.google.com/">Pretrained Models</a> </h3>
</p>

## Introduction
<img src=".\overview.png">
This paper introduces a novel architecture for trajectory-conditioned forecasting of future 3D scene occupancy. In contrast to methods that rely on variational autoencoders (VAEs) to generate discrete occupancy tokens, which inherently limit representational capacity, our approach predicts multi-frame future occupancy in an end-to-end manner directly from raw image features. Inspired by the success of attention-based transformer architectures in foundational vision and language models such as GPT and VGGT, we employ a sparse occupancy representation that bypasses the intermediate bird's eye view (BEV) projection and its explicit geometric priors. This design allows the transformer to capture spatiotemporal dependencies more effectively. By avoiding both the finite-capacity constraint of discrete tokenization and the structural limitations of BEV representations, our method achieves state-of-the-art performance on the nuScenes benchmark for 1‒3 second occupancy forecasting, outperforming existing approaches by a significant margin. Furthermore, it demonstrates robust scene dynamics understanding, consistently delivering high accuracy under arbitrary future trajectory conditioning.

## BibTeX
```
@article{du2025sparseworld,
  title={SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model},
  author={Du, Jiayuan and Zhao, Yiming and Guo, Zhenglong and Pan, Yong and Hou, Wenbo and Hao, Zhihui and Zhan, Kun and Chen, Qijun},
  journal={arXiv preprint arXiv:2511.22039},
  year={2025}
}
```

## Acknowledgements

The project is partially based on some awesome repos: [OPUS](https://github.com/jbwang1997/OPUS), [VGGT](https://github.com/facebookresearch/vggt), and [DrivingForward](https://github.com/fangzhou2000/DrivingForward). Many thanks to these projects for their excellent contributions!
