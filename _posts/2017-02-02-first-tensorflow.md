---
layout: post
title: "First Tensor Flow Project"
date: 2017-02-02 16:28:38 -0700
categories: projects
---
For those that want to skip the preliminaries, go directly to the project page:

[Sherlock Holmes Word Vectors](https://pat-coady.github.io/word2vec/)

## Background

My neural network learning progression followed the typical path. I began by building networks using matrix operations in both Octave and Python. It is very satisfying to build a network from scratch and have it perform respectably on the MNIST digit recognition problem. This is also the best (only?) way to truly understand back-propagation, regularization, gradient descent, weight initialization and different optimization algorithms.

Next I moved to the neural net packages in R and Python ([sklearn](http://scikit-learn.org/stable/) and [Keras](https://keras.io/)). This is the way to go when you're ready to try different data sets, layer sizes, optimizers and so on. With these packages, you can efficiently do hyper-parameter searches and run parallel jobs.

I recently finished Geoff Hinton's Neural Network course on [Coursera](https://www.coursera.org/) (excellent, by the way). The section on using NNs to learn word vectors was fascinating and I wanted to give it a try myself. Around the same time I became curious about TensorFlow. So, it was time to kill two birds with one stone.

