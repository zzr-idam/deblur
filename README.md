# deblur
UHD Image Deblurring via Multi-scale Cubic-Mixer

# abstract
Currently, transformer-based algorithms are making a splash in the domain of image deblurring. Their achievement depends on the self-attention mechanism  with CNN stem to model long range dependencies between tokens. Unfortunately, this ear-pleasing pipeline introduces high computational complexity and makes it difficult to run an ultra-high-definition image on a single GPU in real time. To trade-off accuracy and efficiency, the input degraded image is computed cyclically over three dimensional (C, W, and H) signals without a self-attention mechanism. We term this deep network as Multi-scale Cubic-Mixer, which is acted on both the real and imaginary components after fast Fourier transform to estimate the Fourier coefficients and thus obtain a deblurred image. Furthermore, we combine the multi-scale cubic-mixer with a slicing strategy to generate high-quality results at a much lower computational cost. Experimental results demonstrate that the proposed algorithm performs favorably against the state-of-the-art deblurring approaches on the several benchmarks and a new ultra-high-definition dataset in terms of accuracy and speed.

# cite

@article{zheng2022uhd,

  title={UHD Image Deblurring via Multi-scale Cubic-Mixer},
  author={Zheng, Zhuoran and Jia, Xiuyi},
  
  journal={arXiv preprint arXiv:2206.03678},
  
  year={2022}
  
}
