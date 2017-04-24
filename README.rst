UrbanSim
========

.. image:: https://travis-ci.org/UDST/urbansim.svg?branch=master
   :alt: Build Status (Linux)
   :target: https://travis-ci.org/UDST/urbansim


.. image:: https://ci.appveyor.com/api/projects/status/0ygo756020jpcrg3?svg=true
   :alt: Build Status (Windows)
   :target: https://ci.appveyor.com/project/pksohn/urbansim


.. image:: https://coveralls.io/repos/UDST/urbansim/badge.png?branch=master
   :alt: Test Coverage
   :target: https://coveralls.io/r/UDST/urbansim?branch=master

New version of UrbanSim, a tool for modeling metropolitan real estate
markets

.. image:: http://i.imgur.com/4YyN8ob.jpg
   :alt: UrbanSim

`Detailed documentation <http://udst.github.io/urbansim/>`__ for
UrbanSim is now available.

`Click
here <http://udst.github.io/urbansim/gettingstarted.html#installation>`__
for installation instructions.

Let us know what you are working on or if you think you have a great use case
by tweeting us at ``@urbansim`` or post on the UrbanSim `forum`_.

UrbanSim History
----------------

UrbanSim (http://www.urbansim.com) is a model system for analyzing
urban development. It is an open source platform that has been
continuously refined and distributed for planning applications around
the world for over 15 years. Part of the evolution of the platform is
the necessary process of re-engineering the code to take advantage of
new developments in computational libraries and infrastructure. We
implemented UrbanSim initially in Java in the late 1990's, and by 2005
determined that it was time to re-implement it in Python, and created
the Open Platform for Urban Simulation (OPUS) software implementation at
that time. Now, almost a decade later, it is time again to revisit the
implementation to take advantage of an amazing amount of innovation in
the scientific computing community. The new implementation is hosted on
this GitHub site, and maintained by UrbanSim Inc. and a growing
community of contributors.

New UrbanSim Implementation
---------------------------

This new code base is a streamlined complete re-implementation of the
longstanding UrbanSim project aimed at *reducing the complexity* of
using the UrbanSim methodology. Redesigned from the ground up, the new
library is trivial to install, the development process is made
transparent via this GitHub site, and exhaustive documentation has been
created in the hopes of making modeling much more widely accessible to
planners and new modelers.

We lean heavily on the `PyData <http://pydata.org>`__ community to make
our work easier - Pandas, `IPython <http://ipython.org/>`__, and
`statsmodels <http://statsmodels.sourceforge.net/>`__ are ubiquitous in
this work. These Python libraries essentially replace the UrbanSim
Dataset class, tools to read and write from other storage, and some of
the statistical estimation previously implemented by UrbanSim.

This makes our task easier as we can focus on urban modeling and leave
the infrastructure to the wider Python community. The
`Pandas <http://pandas.pydata.org>`__ library is the core of the new
UrbanSim, which is an extremely popular data manipulation library with a
large community providing support and a very helpful
`book <http://www.amazon.com/Python-Data-Analysis-Wes-McKinney/dp/1449319793>`__.

We have now converted a full set of UrbanSim models to the new
framework, and have running applications for the Paris, Albuquerque,
Denver, Bay Area, and Detroit regions. We have implemented a complete
set of hedonic price models, location choice models, relocation and
transition models, as well as a new real estate development model using
proforma analysis.

We do strongly recommend that you contact the team at www.urbansim.com about your
project to make sure you can get support when you need it,
and know what you are getting into. For major applied projects,
professional support is highly recommended.

Reporting bugs and contributing to UrbanSim
--------------------------------------------

Please report any bugs you encounter via `GitHub Issues <https://github.com/UDST/urbansim/issues>`__.

If you have improvements or new features you would like to see in UrbanSim:

1. Open a feature request via `GitHub Issues <https://github.com/UDST/urbansim/issues>`__.
2. See our code contribution instructions `here <https://github.com/UDST/urbansim/blob/master/CONTRIBUTING.md>`__.
3. Contribute your code from a fork or branch by using a Pull Request and request a review so it can be considered as an addition to the codebase.

Academic literature
-------------------

A selection of academic literature on UrbanSim can be found `here <http://www.urbansim.com/research/>`_.

.. _forum: http://discussion.urbansim.com/