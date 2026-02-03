# SDAR-VL: Stable and Efficient Block-wise Diffusion for Vision-Language Understanding


> Official implementation of **"SDAR-VL: Stable and Efficient Block-wise Diffusion for Vision-Language Understanding"**  

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](./LICENSE)  [![HuggingFace: Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/collections/JetLM/sdar-vl) [![Paper: Arxiv](https://img.shields.io/badge/Technical%20Report-Arxiv-red)](https://arxiv.org/abs/2512.14068)

</div>
sdar
## 🔍 Overview

**SDAR-VL** is the first large-scale **block-wise discrete diffusion** framework for **vision-language understanding (VLU)**.  
It provides a stable and efficient alternative to autoregressive (AR) decoders by introducing an integrated training framework that improves convergence, stability, and efficiency.

Key Features

- 🧩 **Block-wise Diffusion Backbone** — Parallel intra-block denoising with causal inter-block dependencies  
- 🔄 **Asynchronous Block-wise Noise Scheduling** — Diversifies supervision and smooths optimization  
- ⚖️ **Effective Mask Ratio Scaling** — Unbiased loss normalization under stochastic masking  
- 📈 **Progressive Beta Noise Curriculum** — Improves convergence and coverage over training  
- 📊 **SOTA Performance** — Matches or surpasses AR models like *LLaVA-OneVision* under matched setups, and achieves **state-of-the-art results among diffusion-based multimodal models**.


## 🤗 Model Zoo


| Model                 | Type               | Link                                                                 |
|------------------------|--------------------|----------------------------------------------------------------------|
| SDAR-VL-Instruct-4B         | Instruct               | [https://huggingface.co/JetLM/SDAR-VL-Instruct-4B](https://huggingface.co/JetLM/SDAR-VL-Instruct-4B) |
| SDAR-VL-Instruct-8B           | Instruct               | [https://huggingface.co/JetLM/SDAR-VL-Instruct-8B](https://huggingface.co/JetLM/SDAR-VL-Instruct-8B)     |
| SDAR-VL-Think-4B           | Think               | [https://huggingface.co/JetLM/SDAR-VL-Think-4B](https://huggingface.co/JetLM/SDAR-VL-Think-4B)     |
| SDAR-VL-Think-8B      | Think               | [https://huggingface.co/JetLM/SDAR-VL-Think-8B](https://huggingface.co/JetLM/SDAR-VL-Think-8B) |


## ⚙️ Usage

### Inference

```bash
python generate.py
```

### Training

For detailed instructions on how to fine-tune the model on your own dataset, please refer to the guide in the `training` directory: [**training/README.md**](./training/README.md).


## 📬 Contact

For issues or inquiries:

- **Shuang Cheng**, Shanghai AI Lab (chengshuang@pjlab.org.cn)
- **Biqing Qi** (Corrsponding Author), Shanghai AI Lab (qibiqing@pjlab.org.cn)

## 🔬 Citation
```
@misc{cheng2025sdarvl,
  title        = {SDAR-VL: Stable and Efficient Block-wise Diffusion for Vision-Language Understanding},
  author       = {Cheng, Shuang and Jiang, Yuhua and Zhou, Zineng and Liu, Dawei and Wang, Tao and Zhang, Linfeng and Qi, Biqing and Zhou, Bowen},
  year         = {2025},
  note         = {Zhejiang University, Shanghai AI Laboratory, Tsinghua University, Shanghai Jiao Tong University, ByteDance}
}

```