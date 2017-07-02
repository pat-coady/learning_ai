---
layout: post
title: "TensorFlow Speed: Build from Source and Hand-Built GRU"
date: 2017-04-04 12:00:00 -0700
categories: projects
comments: true
---
I recently compared the performance of various RNN cells in a word prediction application. While doing this, I noticed that the Gated Recurrent Unit (GRU) ran slower per epoch than the LSTM cell. Based on the computation graphs for both cells, I expected the GRU to be a bit faster (also confirmed in [literature](https://arxiv.org/pdf/1412.3555v1.pdf)). I launched an investigation into runtimes, including building TensorFlow from source, hand-building a GRU, laptop vs. Amazon EC2 GPU, and feeddict vs. QueueRunner.

Here is a brief summary of the test case I used for all the below benchmarks:

    embedding layer width = 64
    rnn width = 192
    rnn sequence length = 20
    hidden layer width = 96 (fed by final RNN state)
    learning rate = 0.05, momentum = 0.8 (SGD with momentum)
    batch size = 32
    epochs = 2 (~250k examples per epoch)

## Build TensorFlow from Source

I suspect many people ignore these warnings at TensorFlow startup:

{% highlight plaintext %}
The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
{% endhighlight plaintext %}    

The TensorFlow [Performance Guide](https://www.tensorflow.org/performance/performance_guide) recommends building from source, but gives no benchmarks on speed improvement. I rarely go through the effort of building from source, and never on a complex piece of software like TensorFlow. I decided to give it a shot, and honestly, it wasn't very difficult.

The speed improvement was 1.8x on a CPU (noted as 'laptop' in the table):

|---
| | pip (laptop)| source (laptop) | pip (GPU)| source (GPU) |
|-|:-:|:-:|:-:|:-:|:-:|:-:|
|**GRUCell** | 390s | **258s** | 240s | **228s** | 
|**Hand-Built GRU** | 339s | **187s** | 210s | **209s** |
|---

**Table 1.** Comparison of run times. **laptop** = Intel i5 w/ Linux. **GPU** = AWS EC2 p2.xlarge instance (1 x NVIDIA K80). **pip** = TensorFlow install with pip. **source** = Build from source using `bazel`.

The speed improvement on a GPU machine was negligible. The CUDA and cuDNN libraries from NVIDIA are already compiled and optimized for their GPU. 

## Hand-Built vs. GRUCell + dynamic_rnn

As you probably noted above, the hand-built GRU provided an additional performance boost on top of compiling for your native CPU. 

|---
| | CPU | AWS EC2 GPU |
|-|:-:|:-:|:-:|:-:|
|**GRUCell** | 258s | 228s |
|**Hand-Built GRU** | **187s** | **209s** |
|---

**Table 2.** 1.4x improvement by hand-crafting a GRU-based RNN vs. using `tf.contrib.rnn.GRUCell` and `tf.nn.dynamic_rnn`.

Here is the code for a hand-built GRU:

{% highlight python %}
def init_rnn_cell(x, num_cells, batch_size):
    """Initialize variables"""
    i_sz = x.shape[1]+num_cells
    o_sz = num_cells
    with tf.variable_scope('GRU'):
        Wr = tf.get_variable('Wr', (i_sz, o_sz), tf.float32, vsi_initializer)
        Wz = tf.get_variable('Wz', (i_sz, o_sz), tf.float32, vsi_initializer)
        W = tf.get_variable('W', (i_sz, o_sz), tf.float32, vsi_initializer)
        br = tf.get_variable('br', o_sz, tf.float32, one_initializer)
        bz = tf.get_variable('bz', o_sz, tf.float32, one_initializer)
        b = tf.get_variable('b', o_sz, tf.float32, zero_initializer)
        h_init = tf.get_variable('h_init', (batch_size, o_sz), tf.float32, zero_initializer)
    
    return h_init

def cell(x, h_1):
    """GRU, eqns. from: http://arxiv.org/abs/1406.1078"""
    with tf.variable_scope('GRU', reuse=True):
        Wr = tf.get_variable('Wr')
        Wz = tf.get_variable('Wz')
        W = tf.get_variable('W')
        br = tf.get_variable('br')
        bz = tf.get_variable('bz')
        b = tf.get_variable('b')
    
    xh = tf.concat([x, h_1], axis=1)
    r = tf.sigmoid(tf.matmul(xh, Wr) + br)     # Eq. 5
    rh_1 = r * h_1
    xrh_1 = tf.concat([x, rh_1], axis=1)
    z = tf.sigmoid(tf.matmul(xh, Wz) + bz)     # Eq. 6
    h_tild = tf.tanh(tf.matmul(xrh_1, W) + b)  # Eq. 8
    h = z*h_1 + (1-z)*h_tild                   # Eq. 7
    
    return h

# 20-step RNN (config.rnn_size == 20)
s = [init_rnn_cell(embed_out[:, 0, :], config.rnn_size, config.batch_size)]
for i in range(config.num_rnn_steps):
    s.append(cell(embed_out[:, i, :], s[-1]))
{% endhighlight python %}    

Here is the same implementation using `tf.contrib.rnn.GRUCell` and `tf.nn.dynamic_rnn`:

{% highlight python %}
# 20-step RNN (config.rnn_size == 20)
rnn_cell = tf.contrib.rnn.GRUCell(config.rnn_size, activation=tf.tanh)
rnn_out, state = tf.nn.dynamic_rnn(rnn_cell, embed_out, dtype=tf.float32)
{% endhighlight python %}

Once you've built your GRU cell, it is three lines of code to instantiate your custom RNN vs. two lines using `tf.contrib.rnn.GRUCell` and `tf.nn.dynamic_rnn`. Clearly, the off-the-shelf RNN cells are easily configured and less prone to error. TensorFlow is all about reducing "cognitive load" so you can focus on designing your networks. But a 1.4x performance hit is a steep price to pay.

#### What's the Problem: `tf.contrib.rnn.GRUCell` or `tf.nn.dynamic_rnn`? 

I didn't see any issues with the `tf.contrib.rnn.GRUCell` implementation. So I suspected that speed hit was from `tf.nn.dynamic_rnn`. So I ran a quick test comparing the speed running using `tf.nn.dynamic_rnn` to `tf.contrib.rnn.static_rnn`.

- `dynamic_rnn`= 258s 
- `static_rnn`= 202s

So, using the dynamic_rnn accounted for most of the speed difference.

Here is the modified code to use `static_rnn`:

{% highlight python %}
# 20-step RNN (config.rnn_size == 20)
rnn_cell = tf.contrib.rnn.GRUCell(config.rnn_size, activation=tf.tanh)
inputs = [embed_out[:, i, :] for i in range(config.num_rnn_steps)]
rnn_out, state = tf.contrib.rnn.static_rnn(rnn_cell, inputs, dtype=tf.float32)
{% endhighlight python %}

## feeddict vs. QueueRunner

I did a quick check to the speed improvement from using a QueueRunner versus a feeddict. I only ran this experiment on the AWS EC2 GPU instance. The performance improvement was very slight: 

- `QueueRunner`= 222s
- `feeddict`= 228s

It was easy to store all the data on the GPU for this benchmark. So, it isn't surprising that the performance is about the same. I was never able to get the GPU utilization over 60% in any scenario, trying several different queue architectures. Training a convolutional net on image data would surely be a different story.

## Laptop vs. Amazon EC2 GPU

Now, this was quite disappointing, especially since I had been paying for GPU compute time with Amazon. The p2.xlarge instance was faster than my laptop, but this was only because I hadn't compiled TensorFlow for my native CPU microarchitecture. Now it seems my laptop beats out the GPU instance (Table 2).

Again, I'm sure the story would be entirely different with a deep convolutional net (CNN) trained on image data. I'll be turning my attention to deep CNNs soon. 

## Summary

I think the takeaways are simple:

1. Build TensorFlow for your machine
2. If you don't need the features of `dynamic_rnn`, then use `static_rnn` for a speed boost.

As always, I hope you found this post helpful. Please comment with questions, corrections or ideas for future posts.

{% if site.disqus.shortname %}
  {% include disqus_comments.html %}
{% endif %}
