BIRDMAn Model API
====================

BIRDMAn includes several default models for count regression but also supports custom modeling. This involves creating a new Stan file and creating a Model object through the BIRDMAn API.

Default Models
--------------

These are the default models that are included in BIRDMAn. They should be usable with minimal knowledge of Stan and are good general purpose models.

.. autoclass:: birdman.default_models.NegativeBinomial
    :members:
.. autoclass:: birdman.default_models.NegativeBinomialSingle
    :members:
.. autoclass:: birdman.default_models.NegativeBinomialLME
    :members:

Table Model
-----------

You should inherit/instantiate this class if you are building a custom model for estimating parameters of an entire table at once.

.. autoclass:: birdman.model_base.TableModel

Single Feature Model
--------------------

This class is designed for those interested in parallelizing model fitting across multiple features at once. We do not explicitly perform parallelization but rather leave that to the user.

.. autoclass:: birdman.model_base.SingleFeatureModel

Model Iterator
--------------

This is a helper class for fitting the constituent ``SingleFeatureModels`` of a given table. It may be helpful to use this iterator in conjunction with a scheduler or other means of job submission.

.. autoclass:: birdman.model_base.ModelIterator
    :members:

Base Model
----------

This is the abstract class from which all BIRDMAn models derive. Note that this class cannot be instantiated on its own.

.. autoclass:: birdman.model_base.BaseModel
    :members:
