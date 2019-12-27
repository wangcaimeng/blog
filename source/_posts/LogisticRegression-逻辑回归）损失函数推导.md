---
title: LogisticRegression(逻辑回归）损失函数推导
date: 2019-03-05 14:53:16
categories:
    - 机器学习
tags: 
    - 算法推倒
    - 学习笔记
mathjax: true
---

昨天面试遇到的问题，今天整理出来。主要是损失函数的推倒。

<!-- more -->

* 预测函数
$$ \widehat{y} = H(x) = \frac{1}{1+e^{-(\boldsymbol{\mathbf{}\omega }^{T}\boldsymbol{x}+b)}} $$
* 两边取对数，变形：
$$ ln\frac{1-\widehat{y}}{\widehat{y}} = -(\boldsymbol{\mathbf{}\omega }^{T}\boldsymbol{x}+b) $$
* 将$1-\widehat{y}$看作 $y=1$的概率，$\widehat{y}$看作$y=0$的概率，即：
$$ ln\frac{p(y=1\mid \boldsymbol{x})}{p(y=0\mid \boldsymbol{x})} = -(\boldsymbol{\mathbf{}\omega }^{T}\boldsymbol{x}+b) $$
$$ p(y=1\mid \boldsymbol{x}) = \frac{e^{-(\boldsymbol{\mathbf{}\omega }^{T}\boldsymbol{x}+b)}}{1+e^{-(\boldsymbol{\mathbf{}\omega }^{T}\boldsymbol{x}+b)}} = 1-H(x) $$
$$p(y=0\mid \boldsymbol{x}) = \frac{1}{1+e^{-(\boldsymbol{\mathbf{}\omega }^{T}\boldsymbol{x}+b)}}=H(x)$$
*  下面通过最大似然估计求解参数$\boldsymbol{\mathbf{}\omega },b$，即最大化$p( y\mid X,\boldsymbol{\mathbf{}\omega},b)$,这里X表示全部样本,假设总样本数为m，则有:
$$p( y\mid X,\boldsymbol{\mathbf{}\omega},b) = \prod_{i=1}^{m}p(y_{i}\mid x_{i},\boldsymbol{\mathbf{}\omega},b)$$
* 对数似然函数为：
$$ l(\boldsymbol{\mathbf{}\omega},b) = \sum_{i=1}^{m}ln(p(y_{i}\mid x_{i},\boldsymbol{\mathbf{}\omega},b))\\
= \sum_{i=1}^{m}((1-y_{i})ln(p(y_{i}=0\mid x_{i},\boldsymbol{\mathbf{}\omega},b)+y_{i}ln(p(y_{i}=1\mid x_{i},\boldsymbol{\mathbf{}\omega},b))\\
=\sum_{i=1}^{m}((1-y_{i})ln(H(x_{i}))+y_{i}ln(1-H(x_{i})))$$
*最大化 $l(\boldsymbol{\mathbf{}\omega},b)$即最小化$-l(\boldsymbol{\mathbf{}\omega},b)$，是一个凸优化问题，使用梯度下降法或牛顿法求解即可。





