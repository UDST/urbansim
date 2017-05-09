Getting Started
===============

Let us know what you are working on or if you think you have a great use case
by tweeting us at ``@urbansim`` or post on the UrbanSim `forum`_.

Installation
------------

.. note::
   In the instructions below we will direct you to run various commands.
   On Mac and Linux these should go in your standard terminal.
   On Windows you may use the standard command prompt, the Anaconda
   command prompt, or even Git Bash (if you have that installed).

Anaconda
~~~~~~~~

UrbanSim is a Python library that uses a number of packages from the
scientific Python ecosystem.
The easiest way to get your own scientific Python installation is to
install `Anaconda <https://docs.continuum.io/anaconda/>`__,
which contains most of the libraries upon which UrbanSim depends.

UrbanSim
~~~~~~~~

Dependencies
^^^^^^^^^^^^

UrbanSim depends on the following libraries, most of which are in Anaconda:

* `bottle <http://bottlepy.org/docs/dev/index.html>`__ >= 0.12
* `matplotlib <http://matplotlib.org>`__ >= 1.3.1
* `numpy <http://numpy.org>`__ >= 1.8.0
* `orca <https://github.com/UDST/orca>`__ >= 1.1
* `pandas <http://pandas.pydata.org>`__ >= 0.17.0
* `patsy <http://patsy.readthedocs.org/en/latest/>`__ >= 0.3.0
* `prettytable <https://code.google.com/p/prettytable/>`__ >= 0.7.2
* `pyyaml <http://pyyaml.org/>`__ >= 3.10
* `scipy <http://scipy.org>`__ >= 0.13.3
* `simplejson <http://simplejson.readthedocs.org/en/latest/>`__ >= 3.3
* `statsmodels <http://statsmodels.sourceforge.net/stable/index.html>`__ >= 0.8.0
* `tables <http://www.pytables.org/>`__ >= 3.1.0
* `toolz <http://toolz.readthedocs.org/en/latest/>`__ >= 0.7
* `zbox <https://github.com/jiffyclub/zbox>`__ >= 1.2

Extras require:

* `pandana <https://github.com/UDST/pandana>`__ >= 0.1

Latest Release
^^^^^^^^^^^^^^

conda
#####

`conda <https://conda.io/docs/>`__, which comes with Anaconda, is the
easiest way to install UrbanSim because it has binary installers for
all of UrbanSim's hard-to-install dependencies.
First, add the `udst channel <https://anaconda.org/udst>`__
to your conda configuration::

    conda config --add channels udst

Then use conda to install UrbanSim::

    conda install urbansim

To update to a new UrbanSim version use the ``conda update`` command::

    conda update urbansim

pip
###

UrbanSim can also be installed from
`PyPI <https://pypi.python.org/pypi/urbansim>`__
via `pip <https://pip.pypa.io/en/latest/>`__::

    pip install urbansim

When using this method it's best to already have most of the dependencies
installed, otherwise pip will try to download and install things like
NumPy, SciPy, and matplotlib.
If you're using Anaconda you will already have all of the hard-to-install
libraries.

To update to a new release of UrbanSim use the ``-U`` option with
``pip install``::

    pip install -U urbansim

Development Version
^^^^^^^^^^^^^^^^^^^

UrbanSim can be installed from our
`development repository <https://github.com/udst/urbansim>`__
using `pip <https://pip.pypa.io/en/latest/>`__, a Python package manager.
pip is included with Anaconda so you should now be able to open a terminal
and run the following command to install UrbanSim::

    pip install -U https://github.com/udst/urbansim/archive/master.zip

This will download urbansim and install the remaining dependencies not
included in Anaconda.

If you need to update UrbanSim run the above command again.

Developer Install
^^^^^^^^^^^^^^^^^

If you are going to be developing on UrbanSim you will want to fork our
`GitHub repository <https://github.com/udst/urbansim>`_ and clone
your fork to your computer. Then run ``python setup.py develop`` to install
UrbanSim in developer mode. In this mode you won't have to reinstall
UrbanSim every time you make changes.

Reporting bugs and contributing to UrbanSim
-------------------------------------------

Please report any bugs you encounter via `GitHub Issues <https://github.com/UDST/urbansim/issues>`__.

If you have improvements or new features you would like to see in UrbanSim:

1. Open a feature request via `GitHub Issues <https://github.com/UDST/urbansim/issues>`__.
2. See our code contribution instructions `here <https://github.com/UDST/urbansim/blob/master/CONTRIBUTING.md>`__.
3. Contribute your code from a fork or branch by using a Pull Request and
   request a review so it can be considered as an addition to the codebase.

Tools of the Trade
------------------

This page provides a brief introduction to Pandas and Jupyter Notebooks -
two of the key tools that the new implementation of UrbanSim relies on.

Pandas
~~~~~~

`Pandas <http://pandas.pydata.org>`_ is a data science library written in
Python which is an outstanding tool for data manipulation and exploration.
To get started, we recommend Wes McKinney's `10 minute video tour <http://vimeo.com/59324550>`_.

Pandas is similar to a relational database with a much easier API than SQL,
and with much faster performance.  However, it makes no attempt to enable
multi-user editing of data and transactions the way a database would.

The previous implementation of UrbanSim, known as OPUS,
implemented much of this functionality itself in the absence of such robust
libraries - in fact, the OPUS implementation of UrbanSim was started around
2005, while Pandas wasn't developed until 2010.

One of the main motivations for the current implementation of UrbanSim is to
refactor the code to make it simpler, faster, and smaller, while leveraging
terrific new libraries like Pandas that have solved very elegantly some of the
functionality UrbanSim previously had to implement directly.

A Note on Pandas Indexing
~~~~~~~~~~~~~~~~~~~~~~~~~

One very important note about Pandas - the real genius of the abstraction is
that all records in a table are viewed as key-value pairs.  Every table has an
`index <http://pandas.pydata.org/pandas-docs/stable/indexing.html>`_ or a
`multi-index <http://pandas.pydata.org/pandas-docs/stable/advanced.html#advanced>`_
which is used to `align <http://pandas.pydata.org/pandas-docs/stable/basics.html#aligning-objects-with-each-other-with-align>`_
the table on the key for that table.

This is similar to having a `primary key <http://en.wikipedia.org/wiki/Unique_key>`_
in a database except that now you can do mathematical operations with columns.
For instance, you can now take a column from one table and a column from
another table and add or multiply them and the operation will automatically
align on the key (i.e. it will add elements with the same index value).

This is incredibly handy.  Almost all of the benefits of using Pandas come down
to using these indexes in intelligent and powerful ways.  But it's not always
easy to get the functionality exactly right the first time.

**Some general advice about using Pandas: if you have a problem with Pandas,
check your indexes, re-check your indexes, and do it one more time for good
measure.**

A surprising amount of the time when you have bugs in your code, the Pandas
series is not indexed correctly when performing the subsequent operations and
it is not doing what you intend.  You've been warned.

To be clear, the canonical example of using Pandas might be having a parcel
table indexed on parcel id and a building table indexed on building_id,
but with an attribute in the buildings table called parcel_id
(the `foreign key <http://en.wikipedia.org/wiki/Foreign_key>`_).

The tables can be merged using

``pd.merge(buildings, parcels, left_on="parcel_id", right_index=True, how="left")``

You will do this a lot.  If you want a comparison of SQL and pandas, check out
this `series of blog posts <http://www.gregreda.com/2013/01/23/translating-sql-to-pandas-part1/>`_.

Jupyter Notebooks
~~~~~~~~~~~~~~~~~
One of our favorite development tools is `Jupyter Notebook <https://jupyter.org/#about-notebook>`_,
which is perfect for interactively executing small cells of Python code.
We use notebooks a LOT, and they are a wonderful way to avoid the command line
in a cross-platform way.  The notebook is a fantastic tool to develop snippets
of code a few lines at a time, and to capture and communicate higher-level
workflows.

This also makes the notebook a fantastic pedagogical tool - in other words
it's great for demos and communicating both the input and output of cells of
Python code (e.g. `nbviewer <https://nbviewer.jupyter.org/>`_).
Many of the full-size examples of UrbanSim on this site are presented
in notebooks.

In many cases, you can write entire UrbanSim models in the notebook, but this
is not generally considered the best practice.  It's entirely up to you though,
and we are happy to share with you our insights from many hours of developing
and using this set of tools.

The Python flavor of Jupyter notebook uses `IPython <http://ipython.org/>`_,
an interactive Python interpreter that is built on Python that helps when
interfacing with the operating system, profiling, parallelizing, and with many
other technical details.

A Gentle Introduction to UrbanSim
---------------------------------

Background
~~~~~~~~~~

UrbanSim has been an active research project since the late 1990's, and has
undergone continual re-thinking, and re-engineering over the ensuing years,
as documented in many of the `accumulated research papers <http://www.urbansim.com/research/>`_.
Below is a brief, high-level summary of UrbanSim in only a few paragraphs from
a modeling/programmer perspective.  In pseudocode, UrbanSim can be boiled down
to a series of models estimated and then simulated in sequence.::

    for model in models:
        model.estimate(model_configuration_parameters)
    for i in range(NUMYEARSINSIMULATION):
        for model in models:
            model.simulate(model_configuration_parameters)

The set of models varies among the many UrbanSim applications to different
regions, due to data availability and cleanliness, the time and resources
that can be devoted to the project, and specific research questions that
motivated the projects.  The set of models almost always includes at least the
following:

Residential Real Estate Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Hedonic Regression Models** estimate and predict real estate prices for
  different residential building types

* **Location Choice Models** estimate and predict where different types of
  households will choose to live, and are usually segmented by income and
  sometimes by other demographics.  These models are generally coupled with
  relocation models to capture the varying rates of relocation by households
  of different demographics.

* **Transition models** generate new households/persons to match
  *control totals* that specify the growth of households by demographics
  makeup.

Non-residential Real Estate Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Hedonic Regression Models** are analogous to the above except for modeling
  the rent received on non-residential building types.

* **Location Choices Models** are analagous to the above except for modeling
  the location choices of jobs/establishments, and are usually segmented by
  employment sector (and also include relocation rate models).

* **Transition models** generate new jobs/firms to match *control totals* that
  specify the growth of businesses by sector.

Real Estate Development Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some representation of real estate development must be modeled to accurately
represent regional real estate markets.  In UrbanSim there are several options
for modeling the development process, but most users are now moving to the Pro
Forma based modeling approach.

* **Development Project Location Choice Models** are the easiest way to
  represent development, which sample from all recent development projects,
  estimate a model on where development is currently being located, and find
  an appropriate location for a copied development.

* **Pro Forma Developer Models** take the perspective of the developer and
  measures the profitability of a proposed development by predicting the cash
  flows from the predicted rent or sales price in a given submarket and
  comparing these inflows to the anticipated development costs of the project.

  Development will only happen where the predicted rent is high enough to cover
  costs of construction and a moderate profit, and will occur roughly to meet
  demand based on the location choice models and control totals.

  This type of developer model is highly flexible and can account for various
  planning policies including affordable housing, parking requirements,
  subsidies of various kinds, density bonuses, and other similar policies.

  Development regulations such as comprehensive plans and zoning provide
  regulatory constraints on what types of developments and what densities
  can be considered by the model.

It should be noted that many other kinds of models can be included in the
simulation loop as well.  For instance, inclusion of scheduled development
events is a key element to representing known future development projects.

In general, any Python script that reads and writes data can be included to
help answer a specific research question or to model a certain real-world
behavior - models can even be parameterized in JSON or YAML and included in the
standard model set, and an ever-increasing set of functionality will be added
over time.

Specifying Scenario Inputs
--------------------------

Although UrbanSim is designed to model real estate markets,
the *raison d'etre* of UrbanSim is as a scenario planning tool. Regional or
city planners want to understand how their cities will develop in the
presence or absence of different policies or in the context of different
assumptions that they have little or no control over, like economic growth or
migration of households.

In a sense, this style of regional modeling is kind of like retirement
planning, but for cities - will there be enough room for all the households and
jobs if the city grows by 3% every year?  What if it grows by 5%?  10%?
If current zoning policies don't appropriately accommodate that growth,
it's likely that prices will rise, but by how much?  If growth is pushed to
different parts of the region, will there be environmental impacts or an
inefficient transportation network that increases traffic, travel times,
and infrastructure costs?  What will the resulting urban form look like?
Sprawl, Manhattan, or something in between?

UrbanSim is designed to investigate these questions, and other questions like
them, and to allow outcomes to be analyzed as assumptions are changed.  These
assumptions can include, but are not limited to the following.

* *Control Totals* specify in a `simple Excel-based format <models/transrelo.html#control-table-formatting>`_
  the basic assumptions on demographic shifts of households and of sector
  shifts of employment. These files control the transition models and which new
  households and jobs are added to the simulation.

* *Zoning Changes* in the form of scenario-specific density limits such as
  ``max_far`` and ``max_dua`` are `passed to the pro formas <developer/index.html#urbansim.developer.sqftproforma.SqFtProForma.lookup>`_
  when testing for feasibility.  Simple `utility functions <https://github.com/udst/sanfran_urbansim/blob/5b93eb4708fc7ea97f38a497ad16264e4203dbca/utils.py#L29>`_
  are also common to *upzone* certain parcels only if certain policies affect
  them.

* *Fees and Subsidies* may also come in to play by adjusting the feasibility
  of buildings that are market-rate infeasible.  Fees can also be collected on
  profitable buildings and transferred to less profitable buildings,
  as with affordable housing policies.

* *Developer Assumptions* can also be tested, like interest rates,
  the impact of mixed use buildings on feasibility, of density bonuses for
  neighborhood amenities, and of lowering or raising parking requirements.

Using Orca as a simulation framework
------------------------------------

Before moving on, it's useful to describe at a high level how `Orca <https://github.com/udst/orca>`_,
the pipeline orchestration framework built for UrbanSim,
helps solve the problems described thus far in this *getting started* document.

Over many years of implementing UrbanSim models, we realized that we wanted a
flexible framework that had the following features:

* Tables can be registered from a wide variety of sources including databases,
  text files, and shapefiles.
* Relationships can be defined between tables and data from different sources
  can be easily merged and used as a new entity.
* Calculated columns can be specified so that when underlying data is changed,
  calculated columns are kept in sync automatically.
* Data processing *models* can be defined so that updates can be performed with
  user-specified breakpoints, capturing semantic steps that can be mixed and
  matched by the user.

To this end Orca implements this functionality as
`tables <https://udst.github.io/orca/core.html#tables>`_,
`broadcasts <https://udst.github.io/orca/core.html#automated-merges>`_,
`columns <https://udst.github.io/orca/core.html#columns>`_,
and model `steps <https://udst.github.io/orca/core.html#steps>`_
respectively.  We decided to implement these concepts with Python functions and
`decorators <http://thecodeship.com/patterns/guide-to-python-function-decorators/>`_.
This is what is happening when you see the ``@orca.DECORATOR_NAME`` syntax everywhere, e.g.: ::

    @orca.table('buildings')
    def buildings(store):
        return store['buildings']

    @orca.table('parcels')
    def parcels(store):
        return store['parcels']

With the use of decorators you can *register* these concepts with the
simulation engine and deal with one small piece of the simulation at a time -
for instance, how to access data for a certain table, or how to compute a
certain variable, or how to run a certain model.

The objects can then be passed to each other using *injection*, which passes
objects by name automatically into a function.  For instance, assuming the
parcels and buildings tables have previously been registered (as above),
a new column called ``total_units`` on the ``parcels`` table can be defined
with a function which takes the buildings and parcels objects as arguments.
The tables that were registered are now available within the function and can
be used in many other functions as well.::

    @orca.column('parcels', 'total_units')
    def residential_unit_density(buildings, parcels):
        return buildings.residential_units.groupby(buildings.parcel_id).sum() / parcels.acres

If done well, these functions are limited to just a few lines which implement a
very specific piece of functionality, and there will be more detailed examples
in the tutorials section.

Note that this approach is inspired by a number of different frameworks (in
Python and otherwise) such as `py.test <https://docs.pytest.org/en/latest/fixture.html>`_,
`flask <http://flask.pocoo.org/>`_, and even web frameworks like
`Angular <https://docs.angularjs.org/guide/di>`_.

Note that this is designed to be an *extremely* flexible framework.  Models can
be injected into tables, and tables into models, and infinite recursion is
possible (this is not suggested!).  Additionally, multiple kinds of decorators
can be added to the same file so that a piece of functionality can be separated
- for instance, an affordable housing module.  On the other hand, models could
be kept together, columns together, and tables together - the organization is
up to you.  We hope that this flexibility inspires innovation for specific use
cases, but what follows is a set of tutorials that we consider best practices.


.. _forum: http://discussion.urbansim.com/