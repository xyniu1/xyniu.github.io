---
title: 'Comparing artificial neural representations with the brain'
date: 2023-05-09
permalink: /posts/2023/05/comparing-neural-reps
tags:
  - neural reps
  - perception
---

Artificial neural networks have countless forms - they have different architectures, they are trained with different objective functions, they use different parameters and hyperparameters, and with all those fixed, random initializations can lead to drastically distinct results. Although deep networks have made remarkable empirical progress, not much success has been made in understanding and characterizing their representations. One such metric is, how much are they like the brain?

The blog post reviews several notable methods in the literature that characterize the similarity between ANN representations and the brain on different levels: on the level of population response, and on the level of "behavior" - either performance on downstream tasks, or perceptual implications. Some content is emoji-coded 🟦 🧠 ⚪. For each of the method, I will -
1. Translate it from words to mathematical definitions (for me, it is the clearest way to show the method);
2. Analyze the mathematical form and see what this method tries to achieve. One confusing point for me is when and how to mean-centered or normalize the data ⚪;
3. Provide intuition on their pros and cons when applied to large-scale ANN representations 🟦, or noisy neural responses 🧠;

I will not show the detailed steps of solving the mathematical problem, as they are usually easy to be deployed from coding packages, such as python scikit-learn.

Limited to the scope of my knowledge, most of the ANNs are trained in the context of vision such as object recognition and video understanding, although the methods can be generally applied to other fields. The ANNs we consider give static and noise-free representations.

This blog post will be of particular interest to readers who want to relate their ANNs to the brain, as well as experimentalists who want to compare their data with state-of-the-art deep networks. This topic is one of the core components of my own research, therefore, this blog post will be constantly updated as I learn.

Problem setup
------

Let $$X$$ be the responses to a set of images in a source system, and $$Y$$ be the respones in the target system. Without loss of generality, we assume $$X$$ is the ANN responses and $$Y$$ is the neural responses.