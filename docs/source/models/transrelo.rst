.. _transition-relocation:

Transition and Relocation Models
================================

Introduction
------------

Transition models are used to add (via copying) or remove rows from a table
based on prescribed rates or totals.
Relocation models are used to select movers from a table, also based on
prescribed rates.

Control Table Formatting
------------------------

The tables of rates and totals used in transition and relocation models have
a particular format that allows them to specify different values for
different segments of the table. Below is an example table::

    num_cars num_children_min num_children_max relocation_rate
           0              nan                2            0.08
           0                2                4            0.05
           0                4              nan            0.03
           1              nan                2            0.09
           1                2                4            0.04
           1                4              nan            0.05

Each column except the ``relocation_rate`` column describes a filter
on the table to which the rates will be applied. The column names are
inspected to determine the type of the filter and the table column
to which the filter applies.

Column names that end with ``_min`` indicate a "greater than or equal to"
filter. For example, the second row of the above table will have a filter
like ``num_children >= 2``.

Column names that end with ``_max`` indicate a "less than" filter.
For example, the second row of the above table will have a filter like
``num_children < 4``. Notice that the maximum is not inclusive.

Column names that do not end in ``_min`` or ``_max`` indicate an
"equal to" filter. In the above table the first, second, and third rows
will have a filter like ``num_cars == 0``.

``nan`` values indicate filters that do not apply in a given row.

Transition Model Notes
----------------------

Transition models have two components. The main interface is the
:py:class:`~urbansim.models.transition.TransitionModel`, but it
is configured by providing a "transitioner".
Transitioners can make transitions based on different inputs like
growth rates and control totals. Available transitioners are:

* :py:class:`~urbansim.models.transition.GrowthRateTransition`
* :py:class:`~urbansim.models.transition.TabularGrowthRateTransition`
* :py:class:`~urbansim.models.transition.TabularTotalsTransition`

Or you could write and provide your own transitioner.
Transitioners are expected to be callable and take arguments of a
`DataFrame`_ with the data to transition and a year number.
They should return a new data table, the indexes of rows added,
the indexes of rows copied, and the indexes of rows removed.

API
---

Transition API
~~~~~~~~~~~~~~

.. currentmodule:: urbansim.models.transition

.. autosummary::

   GrowthRateTransition
   TabularGrowthRateTransition
   TabularTotalsTransition
   TransitionModel

Relocation API
~~~~~~~~~~~~~~

.. currentmodule:: urbansim.models.relocation

.. autosummary::

   RelocationModel

Transition API Docs
~~~~~~~~~~~~~~~~~~~

.. automodule:: urbansim.models.transition
   :members:

Relocation API Docs
~~~~~~~~~~~~~~~~~~~

.. automodule:: urbansim.models.relocation
   :members:

.. _DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
