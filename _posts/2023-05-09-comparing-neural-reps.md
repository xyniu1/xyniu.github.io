---
title: 'Comparing artificial neural representations with the brain'
date: 2023-05-09
permalink: /posts/2023/05/comparing-neural-reps
tags:
  - neural reps
  - perception
---
Artificial neural networks have countless forms - they have different architectures, they are trained with different objective functions, they use different parameters and hyperparameters, and with all those fixed, random initializations can lead to drastically distinct results. Although deep networks have made remarkable empirical progress, not much success has been made in understanding and characterizing their representations. One such metric is, how much are they like the brain?

The blog post reviews several notable methods in the literature that characterize the similarity between ANN representations and the brain on different levels: on the level of population response, and on the level of "behavior" - either performance on downstream tasks, or perceptual implications. Some content is emoji-coded ✅ 🟦 🧠 ⚪. For each of the method, I will -
1. Translate it from words to mathematical definitions ✅ (for me, it is the clearest way to show the method);
2. Analyze the mathematical form and see what this method tries to achieve. One confusing point for me is when and how to mean-centered or normalize the data ⚪;
3. Provide intuition on their pros and cons when applied to large-scale ANN representations 🟦, or noisy neural responses 🧠;

I will not show the detailed steps of solving the mathematical problem, as they are usually easy to be deployed from coding packages, such as python scikit-learn.

Limited to the scope of my knowledge, most of the ANNs are trained in the context of vision such as object recognition and video understanding, although the methods can be generally applied to other fields. The ANNs we consider give static and noise-free representations.

This blog post will be of particular interest to readers who want to relate their ANNs to the brain, as well as experimentalists who want to compare their data with state-of-the-art deep networks. This topic is one of the core components of my own research, therefore, this blog post will be constantly updated as I learn.

Problem setup
------

Let $\mathbf{X}$ be the responses to a set of images in a source system (e.g., a deep ANN), and $\mathbf{Y}$ be the responses in the target system (e.g., neural responses in macaque IT). $\mathbf{X}$ has size $m\times p$, where $m$ is the number of images, $p$ is the number of artificial neurons; $\mathbf{Y}$ has size $m\times n$, where $n$ is the number of neurons recorded in the experiment. 

<img src="/images/setup.svg">

🧠 While the ANN responses are usually static and deterministic, neural activities embed unique time dynamics and are stochastic, i.e., their responses can be different for the same input. For the first problem, the standard practice is to count the number of spikes during a certain time period after the stimulus onset (for example, 70ms to 170ms after image onset). The second problem is more subtle. For most of the methods we discuss, they average the responses over repeated trials, but some argue the trial variability also plays an important role.

🧠 The number of neurons recorded is usually on the scale of hundreds, and the number of images shown to the neurons is usually on the scale of hundreds or thousands.

🟦 The number of artificial neurons, on the contrary, can be arbitrarily large, depending on the network architecture and the chosen layer. In some cases we would have $p>m$.

We usually need to fit some parameters to quantify similarity. Always split your data into a training set and a test set - find parameters that best fit the training set, and apply these parameters in the test set to get a similarity score.


A bag of linear regression methods
------

One important ANN-brain similarity benchmark, [Brain-Score][1], gives a great example of how to apply linear regression methods to this problem. 

✅ **Linear Regression:** It aims to find $\mathbf{w}_i\in\mathbb{R}^{p}$ for each neuron in the source system that maximizes the correlation between the predicted response

\begin{aligned}
\mathbf{\hat{y}}_i=\mathbf{X}\mathbf{w}_i
\end{aligned}

and the real response of neuron $i$ in $\mathbf{Y}$

\begin{aligned}
r_i=\text{corr}(\mathbf{y}_i, \mathbf{\hat{y}}_i).
\end{aligned}

Then take the median of $r_i$ over all neurons in the target system.

The reason for taking a median is that neural responses usually follow a non-normal distribution. Direct use of linear regression to compare neural representations is seldom favored because:
1. It tends to overfit when the number of neurons is close to or even larger than the number of observations, which is indeed the case with ANNs;
2. It is not robust if the neurons' responses are correlated, which is very common in real neurons within a population;
3. It is not symmetric - switching the dependent and independent variables (the source and the target system) will give a different similarity score.

Therefore, Brain-Score proposes to do some dimension reduction before carrying out linear regression. They choose PLS, partial least squares regression, which has the following definition:

✅ **PLS:** Find vectors $\mathbf{a}_i$ and $\mathbf{b}_i$ and take the projections $\mathbf{u}_i=\mathbf{X}\mathbf{a}_i$, $\mathbf{v}_i=\mathbf{Y}\mathbf{b}_i$ to maximize

\begin{aligned}
\rho_i=\text{cov}(\mathbf{u}_i, \mathbf{v}_i) ,
\end{aligned}

subject to 

\begin{aligned}
\lVert \mathbf{a}_i \rVert=1, \lVert \mathbf{b}_i \rVert=1,
\end{aligned}

and

\begin{aligned}
\mathbf{u}_i \perp \mathbf{u}_j, \mathbf{v}_i \perp \mathbf{v}_j
\end{aligned}

for $i\neq j$.

Note: Some versions of PLS formulate the problem in a matrix decomposition way, but I find this version more intuitive. 

PLS closely resembles CCA, canonical-correlation analysis, and the only difference is that CCA tries to maximize the correlation $\text{corr}(\mathbf{u}_i, \mathbf{v}_i)$ instead of covariance. When one neuron's response is multiplied by a constant larger than $1$, its covariance with other neurons will also increase. Therefore, PLS puts more weight to high-response neurons when computing the similarity score.

🧠 This is a reasonable choice since high-response neurons in the brain usually have high signal-to-noise ratio and are more informative.

There is a perhaps more popular way of doing dimension reduction, PCA, principal component analysis, and the procedure of first applying PCA than linear regression is called principal component regression (PCR). Compared to PLS that find dual projections to maximize the covariance between two systems' responses, PCR  picks out directions that maximize the variance within only one system itself. However, directions where $\mathbf{X}$ is the most spread out are not necessarily correlated with $\mathbf{Y}$, which makes PCR inefficient in maximizing the correlation between two systems. On the other hand, some people favor PCR because they think dimension reduction should be done in an unsupervised way to avoid over-fitting, i.e., reducing the dimension of $\mathbf{X}$ should be done without knowing the value of $\mathbf{Y}$, which is exactly what PCR does. For a detailed comparison between these two methods, see a nice [demo][2] by scikit-learn.

In practice, Brain-Score first applies PCA on ANN responses, then carry out PLS between ANNs and the neural responses (in V4 and IT).

The family of linear regression methods is perhaps the most classical one, and remains an important benchmark. But here is the thing to be cautious of: these methods (except for PLS) are invariant to invertible linear transformations. For one-side projection, defining $\hat{\mathbf{X}}=\mathbf{XM}$, or for dual projections, defining $\hat{\mathbf{X}}=\mathbf{XM}$, $\hat{\mathbf{Y}}=\mathbf{YN}$ for invertible matrices $\mathbf{M}$ and $\mathbf{N}$ will give the same similarity score. One need to think through whether this is the desirable scenario. Moreover, if the network is very wide, i.e., the number of neurons is close to or even larger than the number of observations, then it is easy to find $\mathbf{M}$ and $\mathbf{N}$ to make $\hat{\mathbf{X}}$ and $\hat{\mathbf{Y}}$ very similar, giving a falsely high similarity score.

Centered Kernel Alignment
-----
This section mainly refers to [this paper][5] by Kornblith, Norouzi, Lee, and Hinton.

The methods so far have focused on comparing neurons (or in a more machine learning language, comparing features), i.e., comparing the columns of $\mathbf{X}$ and $\mathbf{Y}$. Alternatively, one can compare images or the rows, which means to compare how different the neurons as a population respond to one image or another. ⚪ Column-center $\mathbf{X}$ (each column has zero mean), then $\mathbf{XX^T}\in\mathbb{R}^{m\times m}$ gives the dot product of population responses between every pair of images, and is a statistics of the source system.

🧠 This is called representational similarity matrix in neuroscience. 

One can compare this statistics between the source system and the target system: first vectorize the similarity matrix, take a dot product, and then normalize.

✅ **Dot product-based similarity:** 
\begin{aligned}
s(\mathbf{X}, \mathbf{Y}) = \frac{\langle\text{vec}(\mathbf{X}\mathbf{X^T}), \text{vec}(\mathbf{Y}\mathbf{Y^T})\rangle}{\lVert \text{vec}(\mathbf{X}\mathbf{X^T})\rVert \lVert \text{vec}(\mathbf{Y}\mathbf{Y^T}) \rVert}=\frac{\lVert \mathbf{X^T}\mathbf{Y} \rVert_F^2}{\lVert \mathbf{X^T}\mathbf{X}\rVert_F\lVert\mathbf{Y^T}\mathbf{Y}\rVert_F}, 
\end{aligned}
<p>
where $\|\cdot\|_F$ is the Frobenius norm.
</p>

Let's look at this similarity score more closely. First, it is invariant to isotropic scaling due to the normalizing denominator, which means $s(\mathbf{X}, \mathbf{Y})=s(\alpha\mathbf{X}, \beta\mathbf{Y})$. Note that it is invariant to scaling of the whole matrix, not of individual rows or columns of $\mathbf{X}$. Second, it is not invariant to all invertible linear transformations, but only to orthogonal matrices $\mathbf{M}$ with $\mathbf{M}\mathbf{M^T}=\mathbf{I}$. It is a more stringent invariance and thus is preferable in the pathological case where the number of neurons is close to the number of images.

<p>
If we write the rows of $\mathbf{X}$ as $\mathbf{x}_i$, then the $(i,j)$ term of $\mathbf{X}\mathbf{X^T}$ can be written as $(\mathbf{X}\mathbf{X^T})_{i,j}=\langle\mathbf{x}_i, \mathbf{x}_j\rangle$, which is a linear kernel. In fact, we can use other kernels to rewrite the similarity matrix. Let $\mathbf{K}_{ij}=k(\mathbf{x}_i, \mathbf{x}_j)$ and $\mathbf{L}_{ij}=l(\mathbf{y}_i, \mathbf{y}_j)$ where $k$ and $l$ are two kernels, ⚪ then if we further make $\mathbf{K}$ and $\mathbf{L}$ column-centered (since they are symmetric matrices, column-centered would mean row-centered), then we can rewrite the similarity index as
</p>

✅ **CKA:**
\begin{aligned}
s(\mathbf{X}, \mathbf{Y})=\frac{\langle\text{vec}(\mathbf{K}), \text{vec}(\mathbf{L})\rangle}{\lVert\text{vec}(\mathbf{K})\rVert\lVert\text{vec}(\mathbf{L})\rVert}=\frac{\lVert\mathbf{K^T}\mathbf{L}\rVert_F}{\lVert\mathbf{K}\rVert_F \lVert\mathbf{L}\rVert_F},
\end{aligned}
<p>
where $\|\cdot\|_F$ is the Frobenius norm. 
</p>

A common nonlinear kernel is the RBF kernel, given by 

\begin{aligned}
k(\mathbf{x}_i, \mathbf{x}_j)=\exp (\frac{-\lVert\mathbf{x}_i-\mathbf{x}_j\rVert_2^2}{2\sigma^2}),
\end{aligned}

with a hyperparameter $\sigma$.

CKA seems to outperform previous methods on characterizing the similarity of ANNs' response across multiple layers. Intuitively, for a feedforward network, consecutive layers should be more similar to each other than layers far away in the stack. However, previous methods fail to reproduce this intuition as shown in the figure below, where similarity should be high close to the diagonal. CKA captures this property well, either with linear or nonlinear kernels, hence better identifies the correspondence between hidden layers in ANNs.

<img src="/images/cka.png" width=500>

(Optional: How is CKA related to CCA?) We have identified two advantages of CKA over the bag of linear regression methods - read the section again if you don't know what those are. Therefore, it may seem that CKA is very different from them. But it's not. CKA with linear kernels is just a weighted CCA under mild assumptions. 

To see that, let's first look at the optimal solution of CCA. If $\mathbf{X}\in\mathbb{R}^{m \times p}$ has full rank $p$, $\mathbf{Y}\in\mathbb{R}^{m \times n}$ has full rank $n$, ⚪ and they are column-centered,

✅ **Optimal solution of CCA:**

\begin{aligned}
R^2_{\text{CCA}} = \frac{\sum \text{eigenvalues of } (\mathbf{X^TX})^{-1/2}(\mathbf{X^TY})(\mathbf{Y^TY})^{-1}(\mathbf{Y^TX})(\mathbf{X^TX})^{-1/2}}{p}.
\end{aligned}

Since the sum of eigenvalues equals to trace, and this big matrix is a matrix square, this measure can be rewritten as

\begin{aligned}
R^2_{\text{CCA}} = \lVert (\mathbf{Y^TY})^{-1/2}(\mathbf{Y^TX})(\mathbf{X^TX})^{-1/2} \rVert_F^2 / p.
\end{aligned}

Take singular decomposition $\mathbf{X}=\mathbf{U}_X\mathbf{\Sigma}_X\mathbf{V}_X$ and $\mathbf{Y}=\mathbf{U}_Y\mathbf{\Sigma}_Y\mathbf{V}_Y$, we have

\begin{aligned}
R^2_{\text{CCA}} = \lVert \mathbf{V}_Y^T\mathbf{U}_Y^T\mathbf{U}_X\mathbf{V}_X \rVert_F^2 / p = \lVert \mathbf{U}_Y^T\mathbf{U}_X \rVert_F^2 / p,
\end{aligned}

since $\mathbf{V}_X$ and $\mathbf{V}_Y$ are orthogonal matrices that preserve norm. Finally, write the $i^{\text{th}}$ eigenvector of $\mathbf{XX^T}$ as $\mathbf{u}_X^i$, we have

<p>
\begin{aligned}
R^2_{\text{CCA}} = \sum_{i=1}^p \sum_{j=1}^n \langle \mathbf{u}_X^i, \mathbf{u}_Y^j\rangle^2 / p.
\end{aligned}
</p>

On the other hand, linear CKA can be expressed as

<p>
\begin{aligned}
\text{CKA}(\mathbf{X X^T}, \mathbf{YY^T}) =\frac{\lVert \mathbf{Y^TX}\rVert_F^2}{\lVert \mathbf{X^TX}\rVert_F\lVert\mathbf{Y^TY}\rVert_F}=\frac{\sum_{i=1}^{p} \sum_{j=1}^{n} {\lambda}_X^i {\lambda}_Y^j \langle\mathbf{u}_X^i, \mathbf{u}_Y^j\rangle^2}{\sqrt{\sum_{i=1}^{p}({\lambda}_X^i)^2} \sqrt{\sum_{j=1}^{n}({\lambda}_Y^j)^2}},
\end{aligned}
</p>

where ${\lambda}_X^i$ and ${\lambda}_Y^j$ are the eigenvalues of $\mathbf{XX^T}$ and $\mathbf{YY^T}$, respectively.

Both of the solutions are weighted sums of the inner product between the left singular vectors of $\mathbf{X}$ and $\mathbf{Y}$. Compared to CCA that puts equal weight on the inner products, CKA puts more weight on directions that explain more variance of $\mathbf{X}$ and $\mathbf{Y}$, on the idea that eigenvectors corresponding to smaller eigenvalues are less important. It has the flavor PCR - both in an unsupervised way, PCR reduces dimensionality while CKA adopts a softer version to put unequal weights on each dimension.

Although our derivation imposes requirement on the rank of $\mathbf{X}$ and $\mathbf{Y}$, CKA does not have those requirement in practice. It can be computed without any matrix decompositions, and are shown to be effective when the number of neurons exceeds the number of images.



Statistical Shape Analysis
-----
A huge pitfall of all the methods discussed above is, they do not always obey triangle inequality, i.e., $d(\mathbf{A}, \mathbf{B})+d(\mathbf{B}, \mathbf{C})$ is not always larger than or equal to $d(\mathbf{A}, \mathbf{C})$. It means they can only be used to compare a pair of networks, but will fall short when it comes to analyzing thousands of networks systematically. Moreover, a lack of a proper distance metric forbids the use of almost all clustering methods, where each network can be represented as a point in the "network space", and can be classified into semantically meaningful groups. 

This is where statistical shape analysis comes in handy, as introduced in [this paper][3] by our very own [Alex Williams][4]. 

This method requires $\mathbf{X}$ and $\mathbf{Y}$ to have the same dimension. Therefore, we need to first find a transformation $\phi$ that makes both of the response matrix have shape $(m, k)$. From now on, we shall assume both $\mathbf{X}$ and $\mathbf{Y}$ act on the same space $\mathbb{R}^{m\times k}$.

To find a proper metric $d$ that allows the comparison of multiple networks at the same time, we first need to define "equality". That is, under what operation $T$, we can have $d(\mathbf{X}, \mathbf{X}T)=0$ for any response matrix $\mathbf{X}$, or intuitively, we consider $\mathbf{X}$ and $\mathbf{X}T$ "the same"? We have seen that in the bag of linear regression methods, any invertible linear transformations $T$ applied on $\mathbf{X}$ will give the same similarity score. In CKA, the similarity metric is invariant to orthogonal transformations. Here, we impose a more strict condition on $T$ and restrict it to be a linear isometry, that is, $T$ is a linear transformation that preserves length. On $\mathbb{R}^{m \times k}$ and $\mathbb{S}^{m \times k}$, linear isometries are necessarily orthogonal, therefore, $T$ belongs to a subgroup of the set of orthogonal transformations $\mathcal{O}(p)=\lbrace\mathbf{Q} \in \mathbb{R}^{k \times k}: \mathbf{Q}^T \mathbf{Q}=\mathbf{I}\rbrace$.

These descriptions give a feasible range of $T$ to be chosen from. For example, we can choose $T$ to be the set of permutation matrices, that is, if we randomly permute the columns of $\mathbf{X}$, we will consider the representation to be identical. This generally makes sense because neurons don't have intrinsic orders. We can alternatively choose $T$ to be the set of rotation matrices, that is, if we randomly rotate the vector of population response by some degree, the representation should be considered unchanged. Rotating the population response can lead to negative values, hence this choice may or may not be acceptable when representing neurons' firing rates. Finally, one can always use the most strict equality and only allow $T$ to be the identity mapping.

Once we find an ideal group of $T$, Alex showed the following two expressions are proper metrics, i.e. they obey triangle inequality.

✅ **Generalized shape metrics:** If we let $\mathcal{G}$ be the group of $T$ that we choose, then

<p>
\begin{aligned}
d(\mathbf{X}, \mathbf{Y})=\min_{T\in\mathcal{G}}\lVert \mathbf{X}-\mathbf{Y}T\rVert
\end{aligned}

or

\begin{aligned}
d(\mathbf{X}, \mathbf{Y})=\min_{T\in\mathcal{G}} \arccos \frac{\langle \mathbf{X}, \mathbf{Y}T \rangle}{\lVert\mathbf{X}\rVert \lVert\mathbf{Y}\rVert}
\end{aligned}
</p>

defines a proper metric, and can be used to evaluate the dissimilarity of multiple representations at the same time.

Immediately we notice that these two forms look very different from the similarity scores we saw before. Previously, the score $r^2$ is mostly defined on covariance matrices, which is a squared form of $\mathbf{X}$, but now the dissimilarity $d$ is directly defined on $\mathbf{X}$ up to transformation $T$. Also, the second definition gives an angular distance instead of a linear one, which ranges from $0$ to $\pi$.

Meanwhile, the flexibility in choosing $\phi$ and $T$ gives a wide range of possibilities of the specific form of $d$. If chosen properly, certain $d$ can have a similar form with the previous methods. Alex showed one example in his paper that if we consider a partial whitening transformation for $\phi$,

\begin{aligned}
\mathbf{X}^{\phi}=\mathbf{C} \mathbf{X}(\alpha \mathbf{I}+(1-\alpha)(\mathbf{X}^T \mathbf{C} \mathbf{X})^{-1 / 2}),
\end{aligned}

where $\mathbf{C}=\mathbf{I}-(1/m)\mathbf{1}\mathbf{1}^T$ is a centering matrix; and choose $T\in\mathcal{G}$ to be the group of rotation matrices, the metric can be rewritten as 

\begin{aligned}
d(\mathbf{X}, \mathbf{Y})=\min_{T \in \mathcal{G}} \arccos \frac{\langle\mathbf{X}^{\phi}, \mathbf{Y}^{\phi} T\rangle}{\lVert\mathbf{X}^{\phi}\rVert \lVert\mathbf{Y}^{\phi}\rVert}=\arccos (\sum_l \rho_l),
\end{aligned}

where ${\rho}_l$ is the singular values of $(\mathbf{X}^\phi)^T(\mathbf{Y}^\phi)/\lVert\mathbf{X}^{\phi}\rVert \lVert\mathbf{Y}^{\phi}\rVert$. While in CCA, $\sum_l \rho_l$ is usually used as the similarity score (this can be derived from the previous section: optimal solution of CCA). The two similarity indices differ by an $\arccos (\cdot)$ operator, but the first one is a proper metric while the second one is not.

The most significant advantage of the shape metric is that it enables the comparison of more than two networks at the same time. If we have the pairwise distance of multiple networks, we can use multidimensional scaling (MDS) to embed each network into a vector of length $L$, which is a predetermined hyperparameter. MDS find those vectors by trying to preserve all these pairwise distances given by all possible pairs of the networks. In the figure below, each point is the response matrix $\mathbf{X}$ of a specific brain area embedding by MDS then visualized in the 2D PC space, where similar colors mean the same anatomical region. It is clear that anatomically related regions indeed have similar responses under the shape metric.


<img src="/images/shape.svg">

To be continued...














[1]: https://www.biorxiv.org/content/10.1101/407007v2
[2]: https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html
[3]: https://openreview.net/forum?id=L9JM-pxQOl
[4]: http://alexhwilliams.info/
[5]: https://arxiv.org/abs/1905.00414
