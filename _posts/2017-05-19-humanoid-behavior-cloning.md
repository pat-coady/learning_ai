---
layout: post
title: "Humanoid Behavior Cloning"
date: 2017-05-18 12:00:00 -0700
categories: projects
comments: true
---
Reinforcement Learning (RL) is what first sparked my interest in AI. It can take a human more than a year to learn to walk and even longer to master running. The challenge here is to teach a simulated human to run in a much shorter amount of time. In this post, I share some fun results and videos from teaching a [MuJoCo](http://www.mujoco.org/)-modeled humanoid to run in the [OpenAI Gym](https://gym.openai.com/).

## Overview

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

The Humanoid-v1 model in OpenAI Gym is a complex physics-based model. The environment receives an action in $$\mathbb{R}^{17}$$ and returns an observation in $$\mathbb{R}^{376}$$. The action vector controls 17 different joints in the humanoid model. 

In behavior cloning, you only watch and then mimic an expert performing the task. You collect observations and expert actions and then train a model using supervised learning. I ran 250 simulations of the expert Humanoid running in the OpenAI Gym environment. I captured all observation-action pairs from these rollouts and trained two models:

1. Linear Regression
2. Single-layer NN

## Results

### Linear Model

There is nothing fancy here. This model just minimizes the mean-squared-error over the observation vectors. The inputs and outputs were all centered and normalized before training. No regularization was required because the model capacity is small compared to the data set size. 

<iframe width="560" height="315" src="https://www.youtube.com/embed/gmj45Lmxhy4?rel=0" frameborder="0" allowfullscreen></iframe>

### Single-Layer Neural Network (NN)

There are several possible explanations for the failure of our first behavior cloning attempt:

1. Policy-model capacity is insufficient. The action needs to be a non-linear function of several observation variables.
2. Distribution mismatch. The learned policy will be slightly different than the expert policy. These errors will accumulate during the state trajectory.
3. Non-Markovian observations. Perhaps the observations do not give us enough information about the environment state. For instance, maybe the body part locations are known, but not their velocity. Our model only takes into account a single observation when choosing an action. We don't consider our previous action or observations.

Let's address the first possibility head-on by implementing a policy model with more capacity. We start with a single-hidden-layer NN. The hidden layer has 54 nodes with tanh activations. The output layer is linear, and the loss function is mean-squared-error. No over-fitting was observed on a validation set, so no regularization was added. How do we do?

<iframe width="560" height="315" src="https://www.youtube.com/embed/HRRxu3YYazQ?rel=0" frameborder="0" allowfullscreen></iframe>

It seems our problems were solved by only addressing item 1. 

I was quite surprised that item 2 above (distribution mismatch) didn't cause a problem. This is why. Our policy actions will be slightly different than the expert actions. Perhaps our humanoid takes a slightly short step. And then another short step. Now her body gets ahead of her feet, and she tumbles down. But we never saw this mistake when observing the expert, so we never learned to recover from getting ahead of our feet.

There are several ways of addressing distribution mismatch. One well-known method is [DAgger](https://arxiv.org/pdf/1011.0686.pdf). The basic idea is to follow your learned policy and annotate it with the expert actions. Then you re-train with this additional data, repeating until you have a good policy. (I implemented DAgger, and it didn't improve the learning rate in this problem significantly.)

Finally, regarding item 3, it doesn't seem our observations are providing insufficient state information. If we were missing critical state information, it would be nearly impossible to succeed without maintaining an internal model of the dynamics (which we aren't doing).

## Conclusion

It is amazing that a simple neural net can learn a policy to keep our humanoid running. Of course, a puff of the wind or a set of stairs would trip our friend up. And she hasn't learned how to get up after falling.

If you haven't tried the OpenAI Gym, I highly recommend it. With OpenAI Gym environments, this project was only a day's effort. The environments all have a consistent API: whether it is Go, an Atari game, or a simulated humanoid. It is easy to visualize the results and gain insight into your learning algorithm. And, if you write your algorithm thoughtfully, you can quickly use it on multiple problems with no changes.

## Acknowledgements

First I'd like to thank UC Berkeley for making *CS294: Deep Reinforcement Learning* available to the world. I completed this  while doing the first assignment in CS294.

I would also like to thank OpenAI for creating and supporting OpenAI gym. I've built a few very simple RL environments from scratch. It takes a lot of time to write and debug an RL environment, and I'd much rather be developing my learning algorithm with that time. 

{% if site.disqus.shortname %}
  {% include disqus_comments.html %}
{% endif %}
