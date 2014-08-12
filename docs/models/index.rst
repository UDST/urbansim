Core Models
===========

Introduction
------------

Most of UrbanSim's models are implemented as specialized classes that
help link `pandas <http://pandas.pydata.org>`_ data structures
with other operations. The model selections include:

* :ref:`statistical-models`

  * A regression model for predicting numeric data
    and a location choice model for matching choosers to their
    likely selections.

* :ref:`transition-relocation`

  * For adding/removing members from a population and choosing movers.

Contents
--------

.. toctree::
   :maxdepth: 2

   statistical
   transrelo
   util
