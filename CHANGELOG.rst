v3.3
====

2026/09/16

First release since v3.2 in May 2020. It restores compatibility with current
versions of Python and the scientific Python stack, modernizes the packaging
and continuous integration, and fixes several long-standing bugs.

* Requires Python 3.10 or later, and is tested on Python 3.10 through 3.14 with
  NumPy 1.26 through 2.x, Pandas 2.2 through 3.x, SciPy 1.10+, Statsmodels 0.14+,
  and Orca 1.8+. Drops support for Python 2 and for Python 3.5 through 3.9
  (#235, #239).
* Removes uses of NumPy and Pandas features that no longer exist, such as
  ``np.int`` and ``Series.iteritems()``, which made v3.2 fail on NumPy 1.24+ and
  Pandas 2 (#231, #232).
* Preserves choice-result dtypes and filtering behavior under Pandas 3, and
  corrects chained assignment in the square-foot pro forma under Pandas 3
  copy-on-write semantics (#239).
* Supports Pandana 0.8 for the network accessibility utilities, now declared as
  an optional ``network`` extra: ``pip install "urbansim[network]"`` (#239).
* Fixes ``Developer.pick()`` failing when ``form`` is None: all the forms in the
  feasibility table now compete on profitability, as intended. Passing a flat
  single-form feasibility table still works (#194).
* Fixes the transition model dropping linked rows (e.g. persons) for a row that
  is both copied and removed in the same transition (#233), and no longer reuses
  the ids of removed linked rows for new ones.
* Preserves index names in tables returned by the transition model (#221).
* Warns when rows match none of the segments in a transition model's
  configuration, since they are silently dropped from the updated table, or
  match more than one segment, since they are duplicated (#207).
* The DataFrame explorer handles float-typed zone ids in map queries (#204).
* Moves package metadata to ``pyproject.toml`` and removes ``setup.py`` (#239).
* Replaces Travis CI and AppVeyor with GitHub Actions continuous integration
  that tests the minimum and current dependency versions on Linux, macOS, and
  Windows, checks code style, validates the built distributions, and builds the
  documentation with warnings as errors (#235, #239). Releases are built,
  verified, and published to PyPI by a GitHub Actions workflow using Trusted
  Publishing (#244).
* ``main`` is now the integration branch; ``dev`` and ``master`` are retired
  (#236).
* Thanks to Paul Waddell for the compatibility, packaging, and CI work; to Juan
  Caicedo and Sol Tadeo, whose earlier NumPy and Pandas compatibility fixes
  were incorporated; to Hana Sevcikova for the developer model fix and the
  transition model reports and fix; and to Scott Bridwell and Stefan Coe for
  the reports.

v3.2
====

2020/05/05

* Improved installation and compatibility
* Support for Pandas 1.0
* Various improvements and bug fixes
* Note that active development of certain UrbanSim components has moved to stand-alone libraries in UDST: Developer, Choicemodels, UrbanSim Templates

v3.1.1
======

2017/5/9

* Updated deprecated `sort` method for Pandas Series and DataFrames

v3.1.0
======

2017/5/8

* Python 3 compatibility
* Updated documentation
* Various improvements and bugfixes

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
