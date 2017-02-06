---
layout: post
title: "First Tensor Flow Project"
date: 2017-02-02 16:28:38 -0700
categories: projects
comments: true
---
I recently completed a project using TensorFlow to learn word vectors from 3 *Sherlock Holmes* books. The code can be easily modified to "read" other books from [Project Gutenberg](https://www.gutenberg.org/). This project also contains some handy routines for exploring the learned word vectors. Here is the project summary:

[Sherlock Holmes Word Vectors](https://pat-coady.github.io/word2vec/)

## Background

My neural network learning progression followed the typical path. I began by building networks using matrix operations in both Octave and Python. It is satisfying to build a neural network from scratch and have it perform respectably on the MNIST digit recognition problem. This is also a great way to cement your understanding of back-propagation, regularization, gradient descent, random weight initialization and different optimization algorithms.

Next I moved to neural net packages in R and Python ([sklearn](http://scikit-learn.org/stable/) and [Keras](https://keras.io/)). This is the way to go when you want to quickly iterate on different data sets, NN architectures, optimizers and so on. These packages make hyper-parameter search easy and can launch parallel jobs.

I recently completed Geoff Hinton's Neural Network course (excellent, by the way) on [Coursera](https://www.coursera.org/). The topic of using NNs to learn word vectors was fascinating, and I wanted to give it a try myself. Around the same time I became curious about TensorFlow. So, it was time to kill two birds with one stone.

## Thoughts on TensorFlow

The TensorFlow API was intuitive and relatively easy to learn. The installation documentation is great, including instructions for installing with GPU support. Another bonus are the terrific tutorials and examples on the TensorFlow site.

If you want ultimate flexibility and control over your neural networks, using TensorFlow is the right tool. I've never used Theano, so I can't offer a comparison on performance or ease of use. But it seems the TensorFlow ecosystem is growing quickly (including support for Android, iOS and Raspberry Pi). If you are machine learning practitioner that uses established neural network archtectures, then I would would recommend using Keras.

{% if page.comments %}
<div id="disqus_thread"></div>
<script>
var disqus_config = function () {
this.page.url = 'https://pat-coady.github.io/projects/2017/02/04/projects/2017/02/02/first-tensorflow.html';
};
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = '//https-pat-coady-github-io.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
{% endif %}
