# MMEA: Entity Alignment for Multi-Modal Knowledge Graphs
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/liyichen-cly/MMEA/issues)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Tensorflow](https://img.shields.io/badge/Made%20with-Tensorflow-orange.svg?style=flat-square)](https://www.tensorflow.org/)
[![Paper](https://img.shields.io/badge/KSEM%202020-PDF-yellow.svg?style=flat-square)](http://home.ustc.edu.cn/~liyichen/assets/files/LiyiChen_KSEM20.pdf)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--030--55130--8__12-lightgrey.svg?style=flat-square)](https://link.springer.com/chapter/10.1007/978-3-030-55130-8_12)  
Model code and datasets for paper "[MMEA: Entity Alignment for Multi-Modal Knowledge Graphs](http://home.ustc.edu.cn/~liyichen/assets/files/LiyiChen_KSEM20.pdf)" published in Proceedings of the 13th International Conference on Knowledge Science, Engineering and Management (KSEM'2020).  

> Entity alignment plays an essential role in the knowledge graph (KG) integration. Though large efforts have been made on exploring the association of relational embeddings between different knowledge
graphs, they may fail to effectively describe and integrate the multimodal knowledge in the real application scenario. To that end, in this paper, we propose a novel solution called Multi-Modal Entity Alignment
(MMEA) to address the problem of entity alignment in a multi-modal view. Specifically, we first design a novel multi-modal knowledge embedding method to generate the entity representations of relational, visual
and numerical knowledge, respectively. Along this line, multiple representations of different types of knowledge will be integrated via a multimodal knowledge fusion module. Extensive experiments on two public
datasets clearly demonstrate the effectiveness of the MMEA model with a significant margin compared with the state-of-the-art methods.

## Dataset
Three public multi-modal knowledge graphs with **relational**, **numerical** and **visual** knowledge from paper "[MMKG: Multi-Modal Knowledge Graphs](https://arxiv.org/abs/1903.05485)", i.e., FB15k, DB15k and YAGO15k.
There are **sameAs** links between FB15k and DB15k as well as between FB15k and YAGO15k, which could be regarded as **alignment** relations. 
[Please click here to download the datasets.](https://github.com/nle-ml/mmkb)

## Code
Our code was implemented by extending the public benchmark [OpenEA](https://github.com/nju-websoft/OpenEA), therefore we only public the model code to avoid repetition. We appreciate the authors for making OpenEA open-sourced.

### **Dependencies**
* Python 3.6
* Tensorflow 1.10
* Numpy 1.16

## Citation
If you use this model or code, please kindly cite it as follows:
```
@inproceedings{chen2020mmea,
  title={MMEA: Entity Alignment for Multi-modal Knowledge Graph},
  author={Liyi Chen and Zhi Li and Yijun Wang and Tong Xu and Zhefeng Wang and Enhong Chen},
  booktitle={International Conference on Knowledge Science, Engineering and Management},
  pages={134--147},
  year={2020},
  organization={Springer}
}
```
***
 **Last but not least, if you have any difficulty or question in implementations, please send your email to liyichencly@gmail.com.**
