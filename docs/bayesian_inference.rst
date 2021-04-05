Bayesian inference
==================

Bayesian inference is a technique to learn about a given population through Bayes' Theorem. Given data, :math:`D`, and parameter(s), :math:`\theta`, Bayes' Theorem applies as follows:

.. math::

    P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}

The component parts of this formula are usually described with the following terms:

* :math:`P(\theta)`: *prior* - our "current" knowledge of :math:`\theta` independent of our data
* :math:`P(D|\theta)`: *likelihood* - how likely is the data given our knowledge of the parameters?
* :math:`P(\theta|D)`: *posterior* - the "updated" knowledge of :math:`\theta` after seeing the data
* :math:`P(D)`: *evidence* - the probability of the data over all values of :math:`\theta`

Our goal is to hone in on values of :math:`\theta` that best fit the data according to some statistical model we specify. Importantly, we do so by providing both the *data generative distribution* and the *prior distribution* of the parameters.

One of the biggest hurdles to Bayesian inference is that computing :math:`P(\theta|D)` is hard and more often than not unattainable via analytic methods. Instead, we can *sample* from the posterior distribution enough to get a reasonable estimate. We can do this because the posterior distribution is proportional to the product of the likelihood and the prior. Notice how in Bayes' Theorem the evidence is not a function of :math:`\theta` (our value of interest). Computing the evidence is infeasible, especially for 'omics datasets, so it is fortunate that we do not need to do so.

To address this, scientists often use Markov Chain Monte Carlo (MCMC) methods. In a nutshell, MCMC uses dependent sampling of a distribution to attempt convergence around some parameter estimate.

Markov Chain Monte Carlo
------------------------

MCMC is the harmonious convergence of two techniques: Markov Chain and Monte Carlo sampling. These two individual components have their own independent uses, but putting them together has been a revolutionary development in statistics and modeling. We will briefly describe each part of this procedure.

Markov Chain
^^^^^^^^^^^^

Markov Chains describe stochastic processes of transitions between a set of states. Importantly, the probability of moving to the next state is *only* dependent on the current state (often referred to as "memoryless"). This means that the previous steps in the chain do not matter for the purposes of any particular step. Markov Chains are very useful in predictive modeling, for example when predicting words in a sentence.

Monte Carlo
^^^^^^^^^^^

Monte Carlo sampling is a technique for sampling from a distribution, often in cases where computing the underlying distribution analytically is not possible. This is regularly the case in high-dimensional space where the analytical solution comprises multiple integrals. Instead, we can sample from this distribution to approximate a given value or values.

For a good explanation of Monte Carlo simulation, see the classic `Buffon's Needle <https://simonensemble.github.io/2018-04/buffon>`_ example for a good demonstration of Monte Carlo simulation for estimating :math:`\pi`.

Sampling from the posterior
---------------------------

Combining these two procedures, MCMC uses multiple independent Markov Chains to sample from the posterior distribution. There are many algorithms to perform this sampling but the one we use in BIRDMAn is the No-U-Turn-Sampler (NUTS) that is used in Stan. When sampling, the individual Markov Chains first "warm up" to reach a stationary distribution. This is important as if the chains are not sufficiently warned, the chains may not converge properly and they will not estimate the parameters of interest properly.

Resources
---------

For a good overview of Bayesian inference, see `this <https://towardsdatascience.com/bayesian-inference-problem-mcmc-and-variational-inference-25a8aa9bce29>`_ article.
