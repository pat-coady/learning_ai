---
layout: post
title: "AI Gym Workout"
date: 2017-07-28 12:00:00 -0700
categories: projects
comments: true
---
In a previous post, I described teaching a humanoid to walk by imitating a fellow humanoid. Here I give my humanoid the capability to learn on its own. The humanoid's only resources are the **Proximal Policy Optimization** algorithm, two randomly initialized neural networks, and a teacher that rewards forward progress. I also train nine other simulated robots to thrive in their environments: a swimming snake, a 4-legged "ant," a reaching robot arm, a hopping robot, and several others. I have uploaded the results to the [OpenAI Gym](https://gym.openai.com/envs#mujoco) evaluation scoreboards, and the algorithm achieves several top scores.

## Introduction

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Trust Region Policy Optimization (TRPO) \[1\] has achieved excellent results in challenging continuous control tasks. The high-level idea is to take steps in directions that improve the policy, while simultaneously not straying too far from the old policy. The constraint to stay "near" the old policy is the primary difference between TRPO and the vanilla policy gradient method.

Making too large a change from the previous policy, especially in high-dimensional, nonlinear environments, can lead to a dramatic decrease in performance. For example, a little forward lean helps running speed, but too much forward lean leads to a crash. It can be difficult to recover after you've taken too large of a policy step, especially if it moves you to a new behavior regime. The result is erratic learning performance.

A naive solution is to take minuscule policy steps. The questions become: How small of a step? And how do you take small steps? Presumably, the policy step size would be controlled by the optimizer learning rate. Problematically, the correlation between learning rates and policy step size depend on many factors, including the model architecture, optimizer algorithm, number of training epochs, and data sample. The learning rate may need to be so small that learning grinds to a halt.

TRPO takes a principled approach to controlling the rate of policy change. The algorithm places a constraint on the average KL-divergence ($$D_{KL}$$) between the new and old policy after each update: $$D_{KL}(\pi_{old} \Vert \pi_{new})$$. The learning algorithm then chooses directions that lead to the biggest policy improvement under a budget not changing the policy beyond the ($$D_{KL}$$) limit. The theoretical basis for this approach is outlined in the paper \[1\].

So, what is Proximal Policy Optimization (PPO)? PPO is an implementation of TRPO that adds $$D_{KL}$$ terms to training loss function. With this loss function in place, we train the policy with gradient descent like a typical neural network. TRPO, on the other hand, is a bit more complicated, requiring a quadratic approximation to the KL-divergence and calculating conjugate gradients. \[1\]

In this post, I walk through the implementation details of PPO. All code is available in the [github repository](https://github.com/pat-coady/trpo). And the results on each of the ten environments can be found on the [OpenAI Gym evaluation boards](https://gym.openai.com/envs/Hopper-v1).

## Objectives

1. Train agents on each of the 10 [OpenAI Gym MuJoCo](https://gym.openai.com/envs#mujoco) tasks using the same algorithm and settings
2. Implement Proximal Policy Optimization (PPO)
3. Implement Generalized Advantage Estimation (GAE)
4. Build a clean and easy-to-understand PPO implementation that can be extended with:  
    a. Distributed PPO (parallel, asynchronous training) \[2\]  
    b. Alternative NN architectures  
    c. Q-prop \[5\]  

## Algorithm

{% highlight python linenos %}
value_function = ValueFunction()  # random initialization
policy = Policy()  # random initialization
environment = Environment()
data = []
for iteration in range(N):
    data_old = data
    data = []
    for episode in range(M):
        states, actions, rewards = environment.run_episode(policy)
        state_values = value_function.predict(states)
        advantages = calc_advantage(rewards, state_values)
        data.append((states, actions, rewards, state_values, advantages))
    policy.update(states, actions, advantages)
    value_function.fit(data, data_old)
{% endhighlight %}
<body><br/><strong>Algorithm 1.</strong> Main training procedure</body>

The main procedure in *Algorithm 1* is straightforward. The loop starting on `line 8` captures M policy roll outs. Sequences of `states`, `actions`, and `rewards` are collected from each roll out and appended to the `data` batch. `Line 10` add value estimates to each visited state from the roll outs. With predicted state-values in hand, we calculate the advantages and add these to the data set in `line 11`. The advantage of a state-action is how much better (or worse) an action performs than the expectation of present policy from the same state.

With states, actions, and advantages for each time step in all the policy rollouts, we update the policy in `line 13`. Finally, we update our value function to reflect our latest data. In `line 14` we use the present data batch and the previous data batch to smooth changes in the value function.

{% highlight python linenos %}
pi_old = model.pi(actions, states, theta)
pi_new = model.pi(actions, states, theta)
for epoch in range(EPOCHS):
    loss1 = -mean(pi_new / pi_old * advantages)
    loss2 = beta * kl_divergence(pi_old, pi_new)
    loss3 = eta * square(max(0, KL_targ - 2 * D_KL))
    total_loss = loss1 + loss2 + loss3
    gradients = gradient(total_loss, theta)
    theta = apply_gradient(theta, gradients, learn_rate)
    pi_new = model.pi(actions, states, theta)
    D_KL = kl_divergence(pi_old, pi_new)
    if D_KL > 4 * KL_targ: break  # early stopping
if D_KL > 2 * KL_targ:
    beta = beta * 1.5
    if beta > 30:
        learn_rate = learn_rate / 1.5
elif D_KL < 0.5 * KL_targ:
    beta = beta / 1.5
    if beta < 1 / 30:
        learn_rate = learn_rate * 1.5
{% endhighlight %}
<body><br/><strong>Algorithm 2.</strong> Policy update (adapted from [2])</body>

*Algorithm 2* details the policy update and is adapted from Heess et al. \[2\] (which they adapted from \[4\]). The key to the algorithm is storing the previous distribution $$\pi_{old}(a\vert s)$$ in `line 1` before updating the policy parameters. With the old policy stored, we can compute $$D_{KL}$$ as we make policy gradient updates.

There are 3 loss terms. The `loss1` term leads to a standard policy gradient update (gradients are not pushed through `pi_old`). The `loss2` term penalizes $$D_{KL}(\pi_{old} \Vert \pi_{new})$$. Finally, `loss3` is a squared hinge-loss term that kicks in when $$D_{KL}$$ exceeds the target value (`KL_targ`). Training performance was substantially improved by adding the `loss3` term. Gradients are then calculated: $$\nabla_{\theta}L(s, a, A, \pi_{new}, \pi_{old})$$, and applied normally using an auto-diff package (e.g TensorFlow) and your favorite optimizer ($$A = advantage$$).

After each gradient update, we calculate $$D_{KL}$$. We use early stopping in `line 12` to catch large $$D_{KL}$$ steps. Early stopping was critical to achieving stable learning.

Beginning on `line 13`, we check our $$D_{KL}$$ versus our target value. If the $$D_{KL}$$ is more than double our $$D_{KL}$$ target (`KL_targ`), we strengthen `loss2` by increasing `beta` by 1.5x. If $$D_{KL}$$ is below half of `KL_targ`, we decrease `beta` by 1.5x (effectively encouraging more exploration).

In addition to dynamically adjusting `beta`, we evaluate learning rate after each training iteration. If `beta` falls outside of $$[0.033, 30]$$, this indicates that `loss2` is not stabilizing $$D_{KL}$$. Typically I found this was because the learning rate was too high or too low. When the learning rate is too high, $$D_{KL}$$ will often exceed `KL_targ` target on the very first training step. When the learning rate was too low, the policy barely moves during a training iteration. Dynamically adjusting the learning rate in this manner was very effective and might be applied to other problems.

### Policy and Value Functions

The policy distribution and value function approximators are implemented with 3-layer NNs. The networks are automatically sized based on the number observation and action dimensions in each [MuJoCo](http://www.mujoco.org/) environment. The InvertedPendulum-v1 environment has only 5 observation dimensions and 1 action dimension. At the other extreme, the Humanoid environment has 377 observation dimensions and 17 action dimensions. All hidden layers use a tanh activation. 

* Value function:
    * hid1 size = observation dimensions x 10
    * hid2 size = geometric mean of hid1 and hid3 sizes
    * hid3 size = 5
* Policy:
    * hid1 size = observation dimensions x 10
    * hid2 size = geometric mean of hid1 and hid3 sizes
    * hid3 size = action dimensions x 10

The policy is modeled by a multivariate Gaussian with a diagonal covariance matrix. The NN parameterizes $$\boldsymbol{\mu}$$, and a separately trained vector parameterizes the diagonal of $$\boldsymbol{\Sigma}$$.


### Generalized Advantage Estimation

While not providing a dramatic improvement, using Generalized Advantage Estimation (GAE) \[3\] did help performance. GAE introduces a parameter $$\lambda$$. Instead of using the total sum of discounted rewards to estimate the advantage, GAE uses a weighted sum of n-step advantages.

* 1-step advantage: $$-V(s_t)+r_t+\gamma V(s_{t+1})$$ (temporal difference)
* 2-step advantage: $$-V(s_t)+r_t+\gamma r_{t+1}+\gamma^2 V(s_{t+2})$$
* 3-step advantage: $$-V(s_t)+r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\gamma^3 V(s_{t+3})$$

GAE weights each n-step advantage by $$\lambda^{n-1}$$ and sums them all together. This geometric weighting reduces the variance of the advantage estimator by decreasing the importance of future rewards. By setting $$\lambda=0$$, you get basic TD learning. With $$\lambda=1$$ you get the vanilla advantage function. See the paper for additional details \[3\].

## Interesting Property of the Policy Gradient

If you derive the policy gradient for a policy modeled by a Gaussian, you end up with the following result:

$$\nabla_{\sigma^2}L(s, a, \mu_a, \sigma_a^2) = \frac{1}{2\sigma_a^2}\left[\frac{(a-u_a)^2}{\sigma_a^2}-1\right]\cdot\hat{A}(s, a)$$

This is probably a well-known result, but I hadn't seen it before doing the derivation myself. As expected, if the advantage is 0, then the policy gradient is 0.

This result also leads to some interesting insights:

1. If an action is exactly $$\pm 1 \sigma$$ from the expected action ($$\mu_a$$), then no update is made.
2. If an action is MORE than $$1\sigma$$ from the expected action AND the advantage of the action is positive, the update will INCREASE the policy variance. The policy was overly certain about an incorrect action.
3. If an action is LESS than $$1\sigma$$ from the expected action AND the advantage of the action is positive, the update will DECREASE the policy variance. The policy was too uncertain about a good action.

These properties lead to learning that naturally decreases exploration and increases exploitation as the policy improves.

## Training

I have posted notebooks with interesting training curves for each environment on my [Gist Page](https://gist.github.com/pat-coady).

### Optimizer

The ADAM optimizer is used to train both the value function and policy networks. I compared the performance of the ADAM optimizer to vanilla SGD and SGD with Nesterov Momentum. For this application, ADAM gave the best training results.

Because the network sizes vary by two orders of magnitude, it was necessary to adjust the learning rate based on network size. Intuitively, the optimizer doesn't "know" how many weights it is changing. It only looks at the benefit of making a small change to each weight in isolation. With tens of thousands of weights, the impact of a single update can be much larger than with hundreds of weights. The chosen heuristic was to scale the learning rate by $$\frac{1}{\sqrt{\left\vert{hid2}\right\vert}}$$.

### Training Settings

**Value Function**

* Squared loss
* 10 epochs per update
* batch size = 256
* include replay of previous batch
* learning rate = $$\frac{1 \times 10^{-2}}{\sqrt{\vert{hid2}\vert}}$$
* random shuffle between epochs
* no regularization

**Policy**

* eta = 50
* beta = 1 (initial value, dynamically adjusted during training)
* kl_targ = 0.003
* 20 epochs, full batch updates
* learning rate = $$\frac{9 \times 10^{-4}}{\sqrt{\vert{hid2}\vert}}$$ (initial value, dynamically adjusted during training)
* no regularization

### Normalize Advantages

Neural networks "appreciate" receiving similarly scaled inputs. The reward scaling between the 10 MuJoCo environments varies by several orders of magnitude. It is safe to normalize the advantages ($$\mu=0, \sigma=1$$) because the learning algorithm is interested only in whether an action is better or worse than average. 

### Observation Scaling and Time Step Feature

Just as with rewards, the scale of observations from the environments varies greatly. Here we need to be careful because the policy is actively learning a mapping from observations to actions. If the scale and offset of the observations are updated every batch, the network is chasing a moving target. To slow scale and offset changes we use an exact running mean and standard deviation over all data points. 

## Experiments

### Performance Comparison to Prior Work

*Table 1* and *Table 2* show a comparison of this work to published results \[5\]. Each agent is allowed to train for 30k episodes (in this work, we only trained the HalfCheetah and Swimmer for 3k episodes). Table 1 shows the number of episodes required cross a predetermined reward threshold. *Table 2* displays the maximum 5 episode average reward reached during training.

|---
| Domain | Threshold | PPO | Q-Prop | TRPO | DDPG | 
|-|:-:|:-:|:-:|:-:|:-:|
| Ant | 3500 | 29740 | **4975** | 13825 | N/A |
| HalfCheetah | 4700 | 1966 | 20785| 26370 | **600** |
| Hopper | 2000 | 2250 | 5945 | 5715 | **965** |
| Humanoid | 2500 | N/A | **14750** | >30000 | N/A |
| Reacher | -7 | 14100 | 2060 | 2840 | **1800** |
| Swimmer | 90 | **137** | 2045 | 3025 | 500 |
| Walker | 3000 | 5700 | 3685 | 18875 | **2125** |
|---

<body><strong>Table 1.</strong> Number of episode before crossing threshold. Results for Q-Prop, TRPO, and DDPG taken from Gu et al. [5]. PPO is this work.<br/><br/></body>

|---
| Domain | PPO | Q-Prop | TRPO | DDPG | 
|-|:-:|:-:|:-:|:-:|
| Ant | 3660 | 3534 | **4239** | 957 |
| HalfCheetah | <sup>*</sup>5363 | 4811 | 4734 | **7490** |
| Hopper | **3897** | 2957 | 2486 | 2604 |
| Humanoid | 1248 | **>3492** | 918 | 552 |
| Reacher | **-4.0** | -6.0 | -6.7 | -6.6 |
| Swimmer | <sup>*</sup>**372** | 103 | 110 | 150 |
| Walker | **7280** | 4030 | 3567 | 3626 |
|---

<body><sup>*</sup><small>only 3k episodes run</small><br/><strong>Table 2.</strong> Max 5-episode average reward during first 30k episodes. Results for Q-Prop, TRPO, and DDPG taken from Gu et al. [5]. PPO is this work.</body>

### Videos

Videos from all ten of my trained agents can be viewed in the MuJoCo section of the [OpenAI Gym](https://gym.openai.com/envs#mujoco) website. I've selected a few interesting videos to include below:

**Video 1.** Early in training this Cheetah flipped on his back, made some forward progress, and collected rewards. So he stuck with this approach and got surprisingly fast. This is an interesting local minimum. With the $$D_{KL}$$ constraint, he'll probably never learn to get back on his feet and run like a normal HalfCheetah.

<iframe width="560" height="315" src="https://www.youtube.com/embed/2-cU-_bdfHQ?rel=0" frameborder="0" allowfullscreen></iframe>

**Video 2.** Here you can watch the Humanoid's progress as it learns to walk. This challenging environment has 377 observation dimensions and 17 action dimensions. Early on it adopts a shuffle which it seems to stick with; However, it was still improving its gait when I terminated training.

<iframe width="560" height="315" src="https://www.youtube.com/embed/Tg0Dyu3iQek?rel=0" frameborder="0" allowfullscreen></iframe>

**Video 3.** It is enjoyable watching this long-legged walker learn to run. After only 25,000 training episodes it moves along quite efficiently.

<iframe width="560" height="315" src="https://www.youtube.com/embed/irkXnpZP89s?rel=0" frameborder="0" allowfullscreen></iframe>

## Future Work

With solid framework built, I hope to explore a few ideas:

* NN architectures more suited to policy representation
* Improved value function:
    * Limit KL divergence between updates, or
    * Larger replay buffer
* Learning a dynamics model for predictive control (MPC)
* Speed: parallelism and improved data feeding
* Transfer learning:
    * Can a humanoid pre-trained on the stand-up task learn to walk in fewer episodes?

## Discussion

The PPO algorithm learned policies for a broad range of continuous control problems. I achieved excellent results by running the **same** algorithm on problems ranging from the basic pole cart to a humanoid with 377 observation dimensions and 17 control dimensions. Across these problems, PPO was not overly sensitive to hyperparameter settings. Finally, the algorithm has the nice property of automatically controlling its exploration rate, increasing or reducing the policy variance as learning proceeds.

I hope readers find the code in the [GitHub repository]() useful for their work. Training an agent on a MuJoCo OpenAI Gym environment is as easy as:

`./train.py InvertedPendulum-v1`

I look forward to receiving suggestions for improvement or even pull requests. I also welcome feedback on this post or even just a quick note if you enjoyed it.

Thanks for reading!

### References

1. [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) (Schulman et al., 2016)
2. [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf) (Heess et al., 2017)
3. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf) (Schulman et al., 2016)
4. [GitHub Repository with several helpful implementation ideas](https://github.com/joschu/modular_rl) (Schulman)
5. [Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic](https://arxiv.org/pdf/1611.02247.pdf) (Gu et al., 2017)

{% if site.disqus.shortname %}
  {% include disqus_comments.html %}
{% endif %}
