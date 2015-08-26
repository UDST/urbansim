v3.0.0
======

2015/8/26

* Remove simulation framework, which has been moved to a separate library
  called `Orca <https://udst.github.io/orca/>`_

v2.0.1
======

* Fix index of summed probabilities

  * https://github.com/udst/urbansim/pull/144

v2.0.0
======

* Renamed Location Choice Models to Discscrete Choice Models

  * https://github.com/udst/urbansim/pull/134
  * We generalized the existing location choice model classes into
    discrete choice models with varying capabilities.
    The ``urbansim.models.lcm`` module has been renamed to
    ``urbansim.models.dcm`` and model classes with ``LocationChoice``
    in their name have been renamed to have ``DiscreteChoice`` instead.
  * New options are available to control the behavior of DCMs:

    * ``probability_mode``: The probability mode can take the values
      ``'single_chooser'`` and ``'full_product'`` (default).
      It controls whether the probabilities used for choosing are calculated
      using a single chooser or separately for every chooser.
      The former is a useful performance optimization when there are
      many alternatives.
    * ``choice_mode``: The choice mode can take the values
      ``'individual'`` (default) and ``'aggregate'``.
      It controls whether choices are made one at a time for each chooser
      or all at once for all choosers.
      The latter is appropriate for something like a LCM
      where an alternative taken by one person is no longer available
      to others.
    * At the group level the ``remove_alts`` option specifies whether to
      remove chosen alternatives from the alternative pool between
      performing choices for segments. ``remove_alts`` defaults to ``False``,
      but should be set to ``True`` for LCMs so that alternatives
      are not made available multiple times.

    The default values for these options are appropriate for fully generalized
    discrete choice models, but will need to be set to their non-default
    values to retain the behavior of the old ``LocationChoice`` classes.

* Memoized function injectables

  * https://github.com/udst/urbansim/pull/138
  * Allows users to define a function injectable that has argument-based
    caching that is tied into the larger caching system.

* Allow sampling of alternatives during prediction

  * https://github.com/udst/urbansim/pull/142
