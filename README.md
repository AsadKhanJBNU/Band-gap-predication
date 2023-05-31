## Prediction of organic material band gaps using graph attention network

The electronic properties and technological value of a material are determined by its band
gap. Band gap is notoriously difficult to compute from basic principles and is computationally
intensive to approximate; thus, their prediction is a challenging but critical problem. Machine
learning has enabled advances in predicting the band gap of organic materials in recent years.
However, improved prediction accuracy is still needed. In this study, we used a graph attention
network to improve the performance of band gap approximation. The graph attention model
predicted band gap of materials with a mean absolute error and root mean square error of 0.26
and 0.37 eV, respectively. The performance of the proposed method was evaluated using k-fold
cross-validation. The performance evaluation measures for bang gap prediction by our model
are significantly better than those of state-of-the-art methods. This model is realistic enough to
allow rapid screening of many organic crystal structures to identify new materials accurately.

## Data flow diagram
![Graphical_Abstract](https://user-images.githubusercontent.com/94437138/206943909-5ff17d92-7307-4b0b-82c4-fd79dffa378b.png)

## Requirements 
Python3, PyTorch, RDKit
     

## Web interface
The proposed model is freely available at: http://nsclbio.jbnu.ac.kr/tools/GraphBG/. This web server
accepts SMILES representations as input and determines the band gap.


## Cite

@article{KHAN2023112063,
title = {Prediction of organic material band gaps using graph attention network},
journal = {Computational Materials Science},
volume = {220},
pages = {112063},
year = {2023},
issn = {0927-0256},
doi = {https://doi.org/10.1016/j.commatsci.2023.112063},
url = {https://www.sciencedirect.com/science/article/pii/S0927025623000575},
author = {Asad Khan and Hilal Tayara and Kil To Chong},
keywords = {Band gap, Deep learning, Graph attention, Organic material, OMDB database},
abstract = {The electronic properties and technological value of a material are determined by its band gap. Band gap is notoriously difficult to compute from basic principles and is computationally intensive to approximate; thus, their prediction is a challenging but critical problem. Machine learning has enabled advances in predicting the band gap of organic materials in recent years. However, improved prediction accuracy is still needed. In this study, we used a graph attention network to improve the performance of band gap approximation. The graph attention model predicted band gap of materials with a mean absolute error and root mean square error of 0.26 and 0.37 eV, respectively. The performance of the proposed method was evaluated using k-fold cross-validation. The performance evaluation measures for bang gap prediction by our model are significantly better than those of state-of-the-art methods. This model is realistic enough to allow rapid screening of many organic crystal structures to identify new materials accurately.}
}
