Negative binomial models
========================

You might notice that we use the negative binomial (NB) model quite a lot in BIRDMAn. Here we will briefly explain why we use this distribution for modelling microbiome data. Do note, however, that BIRDMAn is flexible to custom models implementing other distributions such as Dirichlet-multinomial, Poisson-lognormal, etc.

Count models
------------

When we are working with microbiome sequencing (or really most other 'omics data types), our results are typically *counts*. As such, we can't use simple linear regression (which includes fractions and negative numbers) to model the number of each microbe in each sample. We instead want to use a statistical distribution that is explicitly defined on the set of whole numbers. The simplest of such models is the `Poisson distribution <https://en.wikipedia.org/wiki/Poisson_distribution>`_.

The Poisson distribution has a single parameter, :math:`\lambda` that defines the rate of "events". In this distribution, the mean and variance are the same, with both being equal to :math:`\lambda`. This poses a problem for microbiome data, since the variance is typically much greater than the mean. For more on this, see `McMurdie & Holmes 2014 <https://doi.org/10.1371/journal.pcbi.1003531>`_.

Negative binomial model
-----------------------

Enter the negative binomial model. The negative binomial distribution can be thought of as a Poisson with an allowance for extra variance. This is useful for microbiome data, as we can account for the count nature of the data while accomodating inflated variance.

We use the following parameterization from Stan where :math:`i` represents a sample and :math:`j` represents a microbe.

.. math::

    y_{ij} = \textrm{NegativeBinomial}(\mu_{ij}, \phi_j)

In this model, :math:`mu_{ij}` represents the mean count of this microbe in this sample and :math:`\phi_j` represents the dispersion parameter for this microbe. This model is useful because we can model one or both parameters hierarchically according to a generalized linear model with log-link function.

For example, if we expect that the mean abundance depends on whether the sample is a case or control, we can add a parameter and fit it using BIRDMAn.

.. math::

    \ln(\mu_{ij}) = \beta_{0j} + \beta_{1j} x_i + \ln(\textrm{Depth}_i)

In this equation, :math:`\beta_0` is the *intercept* term. This term is the average (log) control sample proportion. For more on this see `this blogpost <https://github.com/flatironinstitute/q2-matchmaker/issues/24>`_.

Here, :math:`x_i` is a binary vector representing whether sample :math:`i` is a case sample (1) or a control sample (0). Thus, :math:`\beta_1` represents the log-fold change of microbial abundance between cases and controls.

Finally, we account for the fact that sampling depth differs across samples by adding a correction term of the log depth of the modeled sample.

.. note::

    This is a pretty basic equation but we can modify it depending on our experimental design or specific questions. For example we could add more covariates, subject effects, or even model the dispersion as a sample dependent parameter!

Output
------

When we fit this model using BIRDMAn, we get *distributions* of plausible parameter values given our data and priors. For a single microbe, we typically are interested in :math:`\beta_0`, :math:`\beta_1`, and :math:`\phi`. When we fit all of the microbes, we end up with :math:`D` distributions for each of these parameter where D is the total number of microbes in the table.

You can imagine that if we, for example, modeled dispersion as a function of both sample and microbe, we would get :math:`N x D` distributions where :math:`N` is the number of samples in our table.
