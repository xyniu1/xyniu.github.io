---
title: 'Comparing artificial neural representations with the brain'
date: 2023-05-09
permalink: /posts/2023/05/comparing-neural-reps
tags:
  - neural reps
  - perception
---

Artificial neural networks have countless forms - they have different architectures, they are trained with different objective functions, they use different parameters and hyperparameters, and with all those fixed, random initializations can lead to drastically distinct results. Although deep networks have made remarkable empirical progress, not much success has been made in understanding and characterizing their representations. One such metric is, how much are they like the brain?

The blog post reviews several notable methods in the literature that characterize the similarity between ANN representations and the brain on different levels: on the level of population response, and on the level of "behavior" - either performance on downstream tasks, or perceptual implications. Some content is emoji-coded 🛎️ 🟦 🧠 ⚪. For each of the method, I will -
1. Translate it from words to mathematical definitions 🛎️ (for me, it is the clearest way to show the method);
2. Analyze the mathematical form and see what this method tries to achieve. One confusing point for me is when and how to mean-centered or normalize the data ⚪;
3. Provide intuition on their pros and cons when applied to large-scale ANN representations 🟦, or noisy neural responses 🧠;

I will not show the detailed steps of solving the mathematical problem, as they are usually easy to be deployed from coding packages, such as python scikit-learn.

Limited to the scope of my knowledge, most of the ANNs are trained in the context of vision such as object recognition and video understanding, although the methods can be generally applied to other fields. The ANNs we consider give static and noise-free representations.

This blog post will be of particular interest to readers who want to relate their ANNs to the brain, as well as experimentalists who want to compare their data with state-of-the-art deep networks. This topic is one of the core components of my own research, therefore, this blog post will be constantly updated as I learn.

Problem setup
------

Let $\mathbf{X}$ be the responses to a set of images in a source system (e.g., a deep ANN), and $\mathbf{Y}$ be the responses in the target system (e.g., neural responses in macaque IT). $\mathbf{X}$ has size $m\times p$, where $m$ is the number of images, $p$ is the number of artificial neurons; $\mathbf{Y}$ has size $m\times n$, where $n$ is the number of neurons recorded in the experiment. 

🧠 While the ANN responses are usually static and deterministic, neural activities embed unique time dynamics and are stochastic, i.e., their responses can be different for the same input. For the first problem, the standard practice is to count the number of spikes during a certain time period after the stimulus onset (for example, [Brain Score][1] chooses 70ms to 170ms after image onset). The second problem is more subtle, and I'd like to think $\mathbf{Y}$ has an invisible third dimension $t$, which stands for trials. For most of the methods we discuss, they average the responses over repeated trials, but some argue the trial variability also plays an important role.

🧠 The number of neurons recorded is usually on the scale of hundreds, and the number of images shown to the neurons is usually on the scale of hundreds or thousands.

🟦 The number of artificial neurons, on the contrary, can be arbitrarily large, depending on the network architecture and the chosen layer. In some cases we would have $p>m$.

We usually need to fit some parameters to quantify similarity. Always split your data into a training set and a test set, where we search for parameters that best fit the training set, and apply these parameters in the test set to get a similarity score.


A bag of linear regression methods
------

One important ANN-brain similarity benchmark, [Brain Score][1], gives a great example of deploying linear regression methods to this problem. 

🛎️ It aims to find $\mathbf{w}_i\in\mathbb{R}^{p}$ for each neuron in the target system that maximizes the correlation between the predicted response
$$\mathbf{\hat{y}}_i=\mathbf{X}\mathbf{w}_i$$
and the real response
$$r_i=\text{corr}(\mathbf{y}_i, \mathbf{\hat{y}}_i).$$
Then take the median of $r_i$ over all neurons in the target system.

The reason for taking a median is that neural responses usually follow a non-normal distribution. This method suffers from the general limitations of linear regression - it tends to overfit when the number of neurons is close to or even larger than the number of observations, which indeed happens with ANNs. It is not robust if the neurons' responses are correlated, again, normal in ANNs and in a real neural population. Therefore, Brain Score proposes to do some dimension reduction before carrying out linear regression. They choose **PLS**, partial least squares regression, which has the following definition:

🛎️ Find vectors $\mathbf{a}_i$ and $\mathbf{b}_i$ and take the projections $\mathbf{u}_i=\mathbf{X}\mathbf{a}_i$, $\mathbf{v}_i=\mathbf{Y}\mathbf{b}_i$ to maximize
$$\rho_i=\text{cov}(\mathbf{u}_i, \mathbf{v}_i) ,$$
subject to 
$$\|\mathbf{a}_i\|=1, \|\mathbf{b}_i\|=1, $$
and
$$\mathbf{u}_i \perp \mathbf{u}_j, \mathbf{v}_i \perp \mathbf{v}_j$$
for $i\neq j$.

Note: Some versions of PLS formulate the problem in a matrix decomposition way, but I find this version the most intuitive. PLS closely resembles **CCA**, canonical-correlation analysis, and the only difference is that CCA tries to maximize the correlation $\text{corr}(\mathbf{u}_i, \mathbf{v}_i)$ instead of covariance. When one neuron's response is multiplied by a constant larger than $1$, covariance will also increase. Therefore, PLS puts more weight to neurons wigh high responses. 

🧠 This is a reasonable choice since high-response neurons in the brain usually have high signal-to-noise ratio and are more informative.

There is a perhaps more popular way of doing dimension reduction, **PCA**, principal component analysis, and the procedure of first applying PCA than linear regression is called principal component regression (**PCR**). Compared to PLS that find dual projections to maximize the covariance between two systems' responses, PCR only picks out directions that maximize the variance of one system within itself. However, directions where $\mathbf{X}$ is the most spread out are not necessarily correlated with $\mathbf{Y}$, which favors the PLS approach. On the other hand, people argue that dimension reduction should be done in an unsupervised way to avoid over-fitting. For a detailed comparison between these two methods, see a nice [demo][2] by scikit-learn.

In practice, brain-score first applies PCA on ANN responses, then carry out PLS between ANNs and the neural responses (in V4 and IT).









[1]: https://www.biorxiv.org/content/10.1101/407007v2 (Brain Score)
[2]: https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html (demo)