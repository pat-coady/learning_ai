---
layout: post
title: "TensorFlow + Keras on AWS"
date: 2017-02-04 16:28:38 -0700
categories: projects
comments: true
---
## Overview

A few months ago I jumped into a Kaggle competition at the last minute. My laptop didn't have enough memory to handle the data set. And, I wanted more CPUs to throw at the problem. So now I had a great excuse to get myself set up to use Amazon cloud computing.

## Sign up for AWS Account

1. If you haven't already, register for an AWS account.
2. Navigate to AWS services -> EC2.
3. Find [Launch Instance] button and click it.

## Select your Amazon Machine Image (AMI):
Select Unbuntu Server (screen shot)

<div style="border: 1px solid black; padding: 15px;" markdown="1">
![Step 1]({{ site.url }}/assets/AWS-server-setup/AWS_step1.png)
</div>

## Select Instance Type
We will select p2.xlarge instance. This is the lowest tier of GPU servers. Once everything is set up, the instance type can be quickly changed depending on the size of your job.

## Configure Instance Details
You will get a screen similar to this:

<div style="border: 1px solid black; padding: 15px;" markdown="1">
![Step 1]({{ site.url }}/assets/AWS-server-setup/AWS_step3.png)
</div>

Here I have selected all the default options except *Protect against accidental termination*. (You want to stop your instance when you are done, not terminate it.)

EBS-optimized instances provide better data throughput to the SSD storage. (I believe all the p2 instances are EBS optimized by defult - check this.)

## Add Storage
I am starting with 100GB of General Purpose SSD storage. Importantly, I have unclicked the *Delete on Termination* box - this is important. The alternative is to take a snapshot of your machine before terminating your instance. Or, mounting a drive with your software installations. This these are manageable approaches, but a pain I think.

**Note:** This approach costs more money, because your storage is always set aside. The cost is on the order of $0.10/GB/day.

## Configure Security Group
(We skipped *Add Tags* - useful if you have many different instances to keep track of.)

Configuring the security group is an important step for viewing a Jupyter Notebook and working interactively on your intance. Here is a screen shot of the settings you will need:

<div style="border: 1px solid black; padding: 15px;" markdown="1">
![Step 6]({{ site.url }}/assets/AWS-server-setup/AWS_step6.png)
</div>

For extra security, you can set your instance to only accept traffic from your own IP address. The above setting (0.0.0.0/0) lets traffic in from any IP address. (Of course, SSH will have a security key. And your Jupyter Notebook sever can be password protected.)

## Launch
Click the launch button. You will likely get a couple warnings:
1. Your server is open for the world to see
2. This instance type is not part of the free AWS tier

Before you can actually launch you need to set up a private key:

<div style="border: 1px solid black; padding: 15px;" markdown="1">
![AWS Key Pair]({{ site.url }}/assets/AWS-server-setup/AWS_key_pair.png)
</div>

You can name your key whatever you'd like. Then download it. I keep mine in my root directory since I usally launch ssh from there. You must make your security key (AWS_GPU_compute.pem or name you chose) root read-only:

{% highlight console %}
$ chmod 400 AWS_GPU_compute.pem
{% endhighlight console %}	

# Log in to your Instance

You will get a screen like this:

<div style="border: 1px solid black; padding: 15px;" markdown="1">
![Launch Status]({{ site.url }}/assets/AWS-server-setup/AWS_launch_status.png)
</div>

Scroll to the bottom-right and click *View Instances*. This will take you to your EC2 dash board and you should see your instance with a green status bubble. Let's SSH to the instance and make sure everything is working.

<div style="border: 1px solid black; padding: 15px;" markdown="1">
![Dashboard]({{ site.url }}/assets/AWS-server-setup/AWS_EC2_Dashboard.png)
</div>

You will only see 1 instance. I already have a different standard CPU instance that I use for other jobs that are too big for my home laptop. Click the *Connect* button and you will get a handy link that you can cut&paste into your terminal to ssh to your instance. Something like this:

{% highlight console %}
$ ssh -i "AWS_GPU_compute.pem" ubuntu@ec2-35-162-197-182.us-west-2.compute.amazonaws.com
{% endhighlight console %}	

You should get a prompt that looks something like this:

{% highlight console %}
ubuntu@ip-172-31-20-90:~$ python3
{% endhighlight console %}	

{% highlight pycon %}	
Python 3.5.2 (default, Sep 10 2016, 08:21:44)
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> print('Hello, World!')
Hello, World!
>>> quit()
{% endhighlight pycon %}	

Congratulations, but we better do something a bit more useful with the compute power now at our fingertips!

# Install Machine Learning Software

{% highlight console %}
$ sudo apt-get update
$ sudo apt-get install python3-pip
$ pip3 install --upgrade pip
$ sudo apt-get install python3-numpy python3-scipy python3-matplotlib python3-pandas python3-nose
$ sudo apt-get install ipython3 ipython3-notebook
$ sudo pip3 install jupyter
$ sudo pip3 install scikit-sklearn
{% endhighlight console %}	

(You may ask, why not just install Anaconda. From what I can determine, the TensorFlow installation for Anaconda doesn't provide GPU support.)

# Let's get Jupyter Notebook up and running

Build config file, it will be put in ~/.jupyter/jupyter_notebook_config.py. This is where you will put your notebook password (to be generated next).

	jupyter notebook --generate-config

Open python and type following to generate password:

{% highlight console %}
ubuntu@ip-172-31-20-90:~$ python3
{% endhighlight console %}	

{% highlight pycon %}	
Python 3.5.2 (default, Nov 17 2016, 17:05:23)
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from notebook.auth import passwd
>>> passwd()
Enter password:
Verify password:
'sha1:f6354de18ac3:39eblaha707435cblahgfe9a6dddedf67568da07'
>>> quit()
{% endhighlight pycon %}	

Open`~/.jupyter/jupyter_notebook_config.py`in a text editor in a separate window. We are going to make some edits

Cut the 'sha1: ...' string to a clipboard and paste it here in the`jupyter_notebook_config.py`file. *Also, uncomment the line*:

{% highlight python %}
c.NotebookApp.password = 'sha: ...'
{% endhighlight python %}	


Add this line below the first comment block in the `jupyter_notebook_config.py` file:

{% highlight python %}
c = get_config()
{% endhighlight python %}	


Generate a SSL web certificate:

{% highlight console %}
$ openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mykey.key -out mycert.pem
$ mv mycert.pem mykey.key ~/.jupyter
{% endhighlight console %}    

Update these lines in `jupyter_notebook_config.py` file (again, uncommenting):

{% highlight python %}
c.NotebookApp.ip = '*'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.certfile = '/home/ubuntu/.jupyter/mycert.pem'
c.NotebookApp.keyfile = '/home/ubuntu/.jupyter/mykey.key'
c.IPKernelApp.pylab = 'inline' # so matplotlib plots are inline with notebook:wq
{% endhighlight python %}

Finally, a couple quick commands to polish off Jupyter install:

{% highlight console %}
$ pip3 install ipywidgets
$ sudo jupyter nbextension enable --py --sys-prefix widgetsnbextension
{% endhighlight console %}

# Fire Up The Notebook!

{% highlight console %}
$ jupyter notebook
{% endhighlight console %}	

Then connect to it by grabbing the dns from your AWS web console. Select the instance from the console, and click the big **Connect** button above. Cut and paste the public dns address from this pop-up:

<div style="border: 1px solid black; padding: 15px;" markdown="1">
![Connect]({{ site.url }}/assets/AWS-server-setup/AWS_connect.png)
</div>

Put a`https://`at the front and a`:8888`at the end. For example:

	https://ec2-35-165-18-173.us-west-2.compute.amazonaws.com:8888

You'll get a warning about an unsigned certificate (you know, the one you generated a few steps earlier). You can safely ignore this (usually by clicking on advanced option in your browser window). Then you'll see the password page of your Jupyter notebook. Enter the password you generated earlier and you are in.

Do a quick check, perhaps `import numpy as np` and generate a random array and print it.

Now the fun begins ... installing TensorFlow and Keras (both actually much easier than everything we've just done.)

## Install TensorFlow and Keras

(Really for the coming 2 section, your best bet is to go to the TensorFlow installation site. Their directions are excellent, and will stay up-to-date. Instally Keras is a 1-liner.)

{% highlight console %}
$ sudo pip3 install keras
$ sudo pip3 install tensorflow
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc0-cp35-cp35m-linux_x86_64.whl
$ sudo pip3 install --upgrade $TF_BINARY_URL
{% endhighlight console %}

## Install CUDA

Need to get 2 files from nVidia site (versions will obviously change - again, go through installation instructions from Tensor Flow site)

1. cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64.deb
2. cudnn-8.0-linux-x64-v5.1-ga.tgz

{% highlight console %}
$ sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64.deb
$ sudo apt-get update
$ sudo apt-get install cuda

$ tar xvzf cudnn-8.0-linux-x64-v5.1-ga.tgz
$ sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
{% endhighlight console %}

Add this to end of your .bashrc:

{% highlight bash %}
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
{% endhighlight bash %}

{% if page.comments %}
<div id="disqus_thread"></div>
<script>
var disqus_config = function () {
this.page.url = 'https://pat-coady.github.io/projects/2017/02/04/AWS-server-setup.html';
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

