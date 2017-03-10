---
layout: post
title: "RNN Language Model and TensorBoard"
date: 2017-03-09 12:00:00 -0700
categories: projects
comments: true
---
In my previous post I used a [Continous Bag-of-Words (CBOW)](https://arxiv.org/pdf/1301.3781.pdf) model to learn word vectors. For this project I built a RNN language model so I could experiment with RNN cell types and training methods. Also, I recently watched a great TensorBoard demo from the [TensorFlow Dev Summit](https://events.withgoogle.com/tensorflow-dev-summit/). So, I added extensive TensorBoard visualization to this project.

This post is only a brief summary. For a more detailed write-up and project code, go to the [GitHub Project Page](https://pat-coady.github.io/rnn/).

## Background

I felt I had a good understanding of Recurrent Neural Networks (RNNs). But you don't fully understand something until you implement it yourself. I previously built a useful toolbox for loading and processing books from [Project Gutenberg](https://www.gutenberg.org/). So, for this project, I stuck with the task of language modeling. Again, using three *Sherlock Holmes* books.

## RNN Architecture

The Internet is filled with examples character-based RNNs. And, I can understand why: it is amazing to see a computer learn to contruct sentences character-by-character. There is even an [example of a RNN generating C code](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) after training on the Linux code base. The character-based approach also has the practical benefit of only needing a ~40-way softmax (alphabet plus some punctuation). But humans read, write, and think in whole words. So I trained my RNN on sequences of words. 

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![Architecture]({{ site.url }}/assets/rnn_and_tensorboard/rnn_diagram.png)
</div>

The architecture is straightforward:

    1. An embedding layer between one-hot word encoding and RNN
    2. RNN: Basic RNN (no memory), GRU or LSTM cell
    3. The last output of the RNN is picked off and fed to a hidden layer
    4. N-way Softmax (N = vocabulary size)

## TensorBoard

For no good reason I had never tried TensorBoard. I was content to pull results from the graph and use matplotlib. But, as I mentioned above, I saw a great TensorBoard demo at the [TensorFlow Dev Summit](https://events.withgoogle.com/tensorflow-dev-summit/). So I decided to give it a try.

From time to time we're all guilty of hacking at a problem. There is an energy barrier to adding probes to your model, re-running the job and figuring out what is going on. So, we hack at it for a little (too long) and hope we get lucky. TensorBoard makes it easy to be disciplined. It is simple to monitor tensor summaries or custom scalar summaries. Afterward you can compare runs and even examine your graph construction. In fact, you may as well monitor "everything" right from the start - then it is there when you need it.

Let's start by looking at the graph:

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![TensorFlow Graph]({{ site.url }}/assets/rnn_and_tensorboard/graph.png)
</div>
  
It is very easy to explore the heirarchy of the design. Here we zoom into our hidden layer [matrix multiply **W**, add the bias **b** and apply the **tanh()**]:

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![TensorFlow Graph - zoom]({{ site.url }}/assets/rnn_and_tensorboard/graph_zoom.png)
</div>

The graph construction looks correct, now we can examine the training results. As you'd expect, you can overlay multiple training-loss curves. You can zoom and easily display the values at the cursors. There is a nice curve smoothing adjustment. And if you have many runs, there is a regular expression filter to grab just the curves you want.

Here I'm taking a look at batch-loss for various hyperparameter settings:

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![Animated training loss]({{ site.url }}/assets/rnn_and_tensorboard/loss.gif)
</div>

The horizontal axis can be adjusted to reflect CPU time or wall clock time. This highlights cases where a model trains in much fewer batches, but is still ultimately slower (Adam optimization seems to give me this problem).

Let's dig deeper into the results. In this series of loss curves you can see 2 runs got stuck on a plateau:

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![Training loss plateau]({{ site.url }}/assets/rnn_and_tensorboard/train_loss_zoom.png)
</div>

Can we figure out what happened? Lets look at some histograms:

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![rnn_out histograms]({{ site.url }}/assets/rnn_and_tensorboard/rnn_out.png)
</div>

As the histograms layer "towards you" they show the evolution of the training. In the upper-left plot, we can see many of the RNN activations (i.e. tanh outputs) are pegged at +/-1. 

The situation is even worse at the hidden layer. Back-propagation isn't going to be able to make it through this - again, upper-left plot:

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![hid_out histograms]({{ site.url }}/assets/rnn_and_tensorboard/hid_out_hist.png)
</div>

So, how did this happen? TensorBoard gives us a different view of how these activations evolved through training:

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![rnn_out distributions]({{ site.url }}/assets/rnn_and_tensorboard/rnn_out_trajectory.png)
</div>

Each level of shading in these plots represents +/-0.5sigma, +/-1sigma, +/-1.5sigma and then min/max. The picture is starting to come together. It looks like we had exploding gradients at the beginning of training. 3 of the 4 scenarios all exploded. But the learning rate is much lower for the upper-left plot, so it doesn't recover during our training. It probably makes sense to add gradient norm clipping.

So, we've found training parameters that seems to be working well. Let's look at the word embedding this RNN has learned. TensorBoard lets you expore embeddings in 3D:

<div style="border: 1px solid black; display: inline-block; padding: 15px; margin: 15px; margin-left: 0px;" markdown="1">
![t-SNE visualization]({{ site.url }}/assets/rnn_and_tensorboard/t-sne.gif)
</div>

## Wrap-Up

It is easy to underestimate how important a good machine learning visualization tool is. Even a simple model (like I presented here) can fall short in complicated ways. You may not even realize your model is underperforming. TensorBoard dramatically reduces the energy required to dig in and really understand what is going on.

I hope this post inspires you to give both TensorFlow and TensorBoard a try. A short Python Notebook in the [GitHub repository](https://pat-coady.github.io/rnn/) implements everything you see here.

{% if site.disqus.shortname %}
  {% include disqus_comments.html %}
{% endif %}


