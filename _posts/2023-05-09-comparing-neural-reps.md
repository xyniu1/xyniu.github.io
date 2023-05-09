---
title: 'Comparing artificial neural representations with the brain'
date: 2023-05-09
permalink: /posts/2023/05/comparing-neural-reps
tags:
  - neural reps
  - perception
---

Artificial neural networks have countless forms - they have different architectures, they are trained with different objective functions, they use different parameters and hyperparameters, and with all those fixed, random initializations can lead to drastically distinct results. Although deep networks have made remarkable empirical progress, not much success has been made in understanding and characterizing their representations. One such metric is, how much are they like the brain?

The blog post reviews several notable methods in the literature that characterize the similarity between ANN representations and the brain. For each of the method, I will -
1. Translate it from words to mathematical definitions (for me, it is the clearest way to show the method);
2. Analyze the mathematical form and see what this method tries to achieve;
3. Provide intuition on their pros and cons when applied to large-scale ANN representations, or noisy neural responses.

I will not show the detailed steps of solving the mathematical problem, as they are usually easy to be deployed from coding packages, such as python scikit-learn.

This blog post will be of particular interest to readers who want to relate their ANNs to the brain, as well as experimentalists who want to compare their data with state-of-the-art deep networks. This topic is one of the core components of my own research, therefore, this blog post will be constantly updated as I learn.

Headings are cool
======

You can have many headings
======

Aren't headings cool?
------