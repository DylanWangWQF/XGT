# XGT

**XGT: Fast and Secure Decision Tree Training and Inference on GPUs**

This repository contains the source code for the paper **[XGT: Fast and Secure Decision Tree Training and Inference on GPUs](https://www.computer.org/csdl/journal/tq/5555/01/11122265/2965ViLYNnq)**, published in *IEEE Transactions on Dependable and Secure Computing 2025*.

## Overview

XGT, short for **eXpress GPU-based Tree**, is a GPU-accelerated framework for privacy-preserving decision tree training and inference using secure multi-party computation (MPC). XGT is designed to improve the efficiency of MPC-protected decision tree operations by transforming key training and inference procedures into GPU-friendly parallel matrix operations.

## Codebase

XGT is built on top of the GPU MPC platform [Piranha](https://github.com/ucbrise/piranha).

Please refer to the Piranha repository for general setup and installation instructions.

## Configuration

Configuration parameters and file paths are defined in:

`files/samples/localhost_config.json`

For local development and testing on a single machine with multiple GPUs, please refer to the related Piranha instructions:

- [Piranha repository](https://github.com/ucbrise/piranha)
- [Running Piranha on a single machine with multiple GPUs](https://github.com/ucbrise/piranha/issues/1)

## Performance Notes

CUDA programming choices can significantly affect the overall performance of MPC operators. In particular, the use of suitable CUDA/Thrust iterators can improve the efficiency of data movement and parallel operations (MPC building blocks) such as fill and copy.

For users who want to build on this repository or develop similar GPU-based MPC projects, we recommend first understanding and experimenting with these CUDA programming techniques.

A related discussion can be found here:

- [CUDA Thrust iterator discussion on Stack Overflow](https://stackoverflow.com/questions/74190787/cuda-thrust-iterator-how-to-use-iterator-to-implement-efficient-fill-and-copy-o)

## Citation

If you use this repository or find our work useful, please cite the paper:

```
@article{wang2025xgt,
  title={XGT: Fast and Secure Decision Tree Training and Inference on GPUs},
  author={Wang, Qifan and Cui, Shujie and Zhou, Lei and Dong, Ye and Bai, Jianli and Koh, Yun Sing and Russello, Giovanni},
  journal={IEEE Transactions on Dependable and Secure Computing},
  volume={22},
  number={6},
  pages={7479--7494},
  year={2025},
  publisher={IEEE},
  doi={10.1109/TDSC.2025.3597394}
}
```

## Disclaimer

This open source project is for proof of concept purposes only and should not be used in production environments. 
