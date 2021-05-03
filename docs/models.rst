BIRDMAn Model Object
====================

Default Models
--------------

These are the default models that are included in BIRDMAn. They should be usable with minimal knowledge of Stan and are good general purpose models.

.. autoclass:: birdman.default_models.NegativeBinomial
    :members:
.. autoclass:: birdman.default_models.NegativeBinomialLME
    :members:
.. autoclass:: birdman.default_models.Multinomial
    :members:

Regression Model
----------------

You should inherit/instantiate this class if you are building a custom regression model with a design matrix.

.. autoclass:: birdman.model_base.RegressionModel
    :members:

Base Model
----------

This is the base model of BIRDMAn used for more complicated models. Primarily for advanced users.

.. autoclass:: birdman.model_base.BaseModel
    :members:
