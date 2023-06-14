# MedIC-DeepCNN: Optimizing Medical Image Classification with Ensemble Learning and Deep Convolutional Neural Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6457912.svg)](https://doi.org/10.5281/zenodo.6457912)
[![shield_license](https://img.shields.io/github/license/pmcafo/MedIC-DeepCNN)](https://www.gnu.org/licenses/gpl-3.0.en.html)

Novel and high-performance medical image classification pipelines are heavily utilizing ensemble learning strategies. The idea of ensemble learning is to assemble diverse models or multiple predictions and, thus, boost prediction performance. However, it is still an open question to what extend as well as which ensemble learning strategies are beneficial in deep learning based medical image classification pipelines.  

In this work, we proposed a reproducible medical image classification pipeline (MedIC-DeepCNN) for analyzing the performance impact of the following ensemble learning techniques: Augmenting, Stacking, and Bagging. The pipeline uses prestigious preprocessing and image augmentation techniques as well as 9 deep convolutional neural network architectures. We utilized it on multiple popular medical imaging datasets with varying complexity. Additionally, we analyzed 12 pooling functions for combining multiple predictions ranged from simple statistical functions like unweighted averaging to more complex learning-based functions like support vector machines.  

![theory](docs/figure.theory.png)

We concluded that the integration of