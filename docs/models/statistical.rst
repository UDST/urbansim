.. _statistical-models:

Statistical Models
==================

Introduction
------------

UrbanSim has two sets of statistical models: regressions and
discrete choice models.
Each has a three stage usage pattern:

#. Create a configured model instance. This is where you will supply
   most of the information to the model such as the actual definition
   of the model and any filters that restrict the data used during
   fitting and prediction.
#. Fit the model by supplying base year data.
#. Make predictions based on new data.

Model Expressions
-----------------

Statistical models require specification of a "model expression" that
describes the model as a mathematical formula.
UrbanSim uses `patsy`_ to interpret
model expressions, but UrbanSim gives you some flexibility
as to how you define them.

`patsy`_ works with string formula like this simplified regression example
(names refer to columns in the `DataFrames`_ used during fitting and prediction)::

    expr = 'np.log1p(sqft_price) ~ I(year_built < 1940) + dist_hwy + ave_income'

In UrbanSim that same formula could be expressed in a dictionary::

    expr = {
        'left_side': 'np.log1p(sqft_price)',
        'right_side': ['I(year_built < 1940)', 'dist_hwy', 'ave_income']
    }

Formulae used with location choice models have only a right hand side
since the models do not predict new numeric values. Right-hand-side formulae
can be written as lists or dictionaries::

    expr = {
        'right_side': ['I(year_built < 1940)', 'dist_hwy', 'ave_income']
    }

    expr = ['I(year_built < 1940)', 'dist_hwy', 'ave_income']

Expressing the formula as a string is always an option.
The ability to use lists or dictionaries are especially useful to make
attractively formatted formulae in :ref:`YAML config files <yaml-config>`.

.. _yaml-config:

YAML Persistence
----------------

UrbanSim's regression and location choice models can be saved as
`YAML <http://en.wikipedia.org/wiki/YAML>`_ files and loaded again
at another time.
This feature is especially useful for estimating models in one location,
saving the fit parameters to disk, and then using the fitted model for
prediction somewhere else.

Use the ``.to_yaml`` and ``.from_yaml`` methods to save files to disk
and load them back as configured models.
Here's an example of loading a regression model, performing fitting, and
saving the model back to YAML::

    model = RegressionModel.from_yaml('my_model.yaml')

    model.fit(data)

    model.to_yaml('my_model.yaml')

You can, if you like, write your model configurations entirely in
YAML and load them into Python only for fitting and prediction.

API
---

Regression API
~~~~~~~~~~~~~~

.. currentmodule:: urbansim.models.regression

.. autosummary::

   RegressionModel
   SegmentedRegressionModel
   RegressionModelGroup

Discrete Choice API
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: urbansim.models.dcm

.. autosummary::

   MNLDiscreteChoiceModel
   SegmentedMNLDiscreteChoiceModel
   MNLDiscreteChoiceModelGroup

Regression API Docs
~~~~~~~~~~~~~~~~~~~

.. automodule:: urbansim.models.regression
   :members:

Discrete Choice API Docs
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: urbansim.models.dcm
   :members:

.. _patsy: http://patsy.readthedocs.org/
.. _DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
.. _DataFrames: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
