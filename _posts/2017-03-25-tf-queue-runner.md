---
layout: post
title: "TensorFlow QueueRunners"
date: 2017-03-25 12:00:00 -0700
categories: projects
comments: true
---

Using QueueRunners is the proper way to keep a hungry TensorFlow model fed. You don't want your high-performance compute resources waiting for data. QueueRunners are launched as independent threads. They look after themselves: refilling when they get low on data and closing when their input source has no more examples. I've built 2 notebooks to demonstrate the use of QueueRunners.

## Overview

It took me some time to understand how Queues worked in TensorFlow. Once I had it figured out, I decided to build 2 simple examples that I could refer to down the road:

- [Example 1](https://gist.github.com/pat-coady/567d30961616f8ec868421eefe6c4a1a): feed queue from variable in graph
- [Example 2](https://gist.github.com/pat-coady/b3e322c2431b7550075138177b70c4f5): feed queue directly from files

I've purposefully stuffed these examples with queues. The first example has 3 queues in series, and the second example has 4 queues. This is to illustrate connections between various types of queues.

## Key Points

### Queues vs. QueueRunners

Each Queue is managed by a QueueRunner. The QueueRunner constructor (i.e. the \_\_init\_\_ method) is given a queue to manage, an enqueue operation to fill the queue, and a close operation. When the QueueRunner gets an OutOfRangeError exception during an enqueue, it will catch this exception and close its queue. The queue will continue to be emptied after being closed, but it won't attempt any more enqueue operations.

As the pipeline of queues are exhausted, the queues throw OutOfRangeError exceptions in sequence. Each succeeding QueueRunner catches the exception and closes its own queue. Of course, the **final** QueueRunner isn't followed by another, so its OutOfRangeError exception isn't caught. We catch this final exception with our try/except/finally block and end the training.

### Pre-built vs. Custom QueueRunners

There are several ready-to-use QueueRunners. Here are the ones I used in the example notebooks:

	- tf.train.input_producer()
	- tf.train.TextLineReader()
	- tf.train.string_input_producer()
	- tf.train.batch()

All of the above automatically create a queue, add a QueueRunner to the graph, and add the QueueRunner to the QUEUE_RUNNERS collection. The QUEUE_RUNNERS collection is used by this statement:

	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

You can also implement a custom Queue and QueueRunner. In the example notebooks I added a FIFOQueue. Going this route, you manually add the QueueRunner to the graph and also to the QUEUE_RUNNERS collection. See the notebooks for details.

### num_epochs

Often the first producer in your pipeline will have a "num_epochs" argument. This controls how many times the producer will completely cycle through its filenames, slices of tensors, range of numbers, etc.

Using the num\_epochs argument adds a local\_variable to the graph which must be initialized. You will see this in the example notebooks. At the end of the epoch, the producer will thrown an OutOfRangeError exception, and trigger the chain of events described earlier.

The alternative is not to pass the num_epochs argument. The producer will loop forever. Instead you will control the number of batches with your training loop (monitoring one or more variables).

### Key Code

Here is the key code followed by explanations:

{% highlight python linenos %}
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)        
try:
    while True:
        tot = sess.run([update_total])
except tf.errors.OutOfRangeError as e:
    coord.request_stop(e)
finally:
    coord.request_stop()
    coord.join(threads)      
{% endhighlight python %}	        

	1. Coordinator to manage QueueRunner threads
	2. Start the QueueRunners (from QUEUE_RUNNERS collection)
	3. Try/except block to catch OutOfRangeError exception
	4. Loop of your design
	5. Typically a call to a training operation
	6. Stop all the threads on OutOfRangeError (i.e. num_epochs complete)
	7. Stop all threads
	9. Stop all threads
	10. Wait until coordinator sees all threads are stopped

## Wrap-Up

I hope these examples help get you up and running using TensorFlow Queues. Please post below if anything is confusing or incorrect.


{% if site.disqus.shortname %}
  {% include disqus_comments.html %}
{% endif %}
