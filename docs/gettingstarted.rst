Getting Started
===============

Installation
------------

Anaconda
~~~~~~~~

UrbanSim is a Python library that uses a number of packages from the
scientific Python ecosystem.
The easiest way to get your own scientific Python installation is to
install `Anaconda <http://docs.continuum.io/anaconda/index.html>`_,
which contains most of the libraries upon which UrbanSim depends.

UrbanSim
~~~~~~~~

UrbanSim can be installed from our
`development repository <https://github.com/synthicity/urbansim>`_
using `pip <https://pip.pypa.io/en/latest/>`_, a Python package manager.
pip is included with Anaconda so you should now be able to open a terminal
and run the following command to install UrbanSim::

    pip install -U https://github.com/synthicity/urbansim/archive/master.zip

This will download urbansim and install the remaining dependencies not
included in Anaconda.

If you need to update UrbanSim run the above command again.

Tools of the Trade
------------------

This page provides a brief introduction to Pandas and IPython Notebooks  - two of the key tools that the new implementation of UrbanSim relies on.

Pandas
~~~~~~

`Pandas <http://pandas.pydata.org>`_ is a data manipulation library written in Python which is an outstanding tool for data manipulation and exploration.  To get started, we recommend Wes McKinney's `10 minute video tour <http://vimeo.com/59324550>`_.

Pandas is similar to a relational database with a much easier API than SQL, and is has much faster performance.  However, it does not enable multi-user editing of data and transactions the way a database would.

The previous implementation of UrbanSim, known as `OPUS <http://urbansim.org>`_, implemented much of this functionality itself in the absence of such robust libraries - in fact, the OPUS implementation of UrbanSim was started around 2005, while Pandas didn't come on the scene until 2010.

One of the main motivations for the current implementation of UrbanSim is to refactor the code to make it simpler, faster, and smaller, while leveraging terrific new libraries like Pandas that have solved very elegantly some of the functionality UrbanSim previously had to implement directly.

A Note on Pandas Indexing
~~~~~~~~~~~~~~~~~~~~~~~~~

One very import note about Pandas - the real genius of the abstraction is that all records in a table are viewed as key-value pairs.  Every table has an index or a multi-index which is used to **align** the table on the key for that table.

This is similar to having a primary key in a database except that now you can do mathematical operations with columns.  For instance, you can now take a column from one table and a column from another table and add or multiply them and the operation will automatically align on the key (i.e. it will add elements with the same index value).

This is incredibly handy.  Almost all of the benefits of using Pandas come down to using these indexes in intelligent and powerful ways.  But it's not always easy to get it right the first time.

**Some general advice about using Pandas: if you have a problem with Pandas, check your indexes, re-check your indexes, and do it one more time for good measure.**

A surprising amount of the time when you have problems, the Pandas series is not indexed correctly when performing the next operation and it is not doing what is intended.  You've been warned.

To be clear, the canonical example might be having a parcel table indexed on parcel id and a building table indexed on building_id, but with an attribute in the buildings table called parcel_id (the foreign key).

The tables can be merged using

``pd.merge(buildings, parcels, left_in="parcel_id", right_index=True, how="left")``

You will do this a lot.  If you want a comparison of SQL and pandas, check out this `series of blog posts <http://www.gregreda.com/2013/01/23/translating-sql-to-pandas-part1/>`_.

IPython
~~~~~~~
`IPython <http://ipython.org/>`_ is an interactive Python interpreter that is built on Python that helps when interacting with the operating system, profiling, parallelizing, and with many other things.

One of the best features of IPython is the `IPython notebook <http://ipython.org/notebook.html>`_, which is perfect for interactively executing small cells of Python code.

This also makes the notebook a fantastic pedagogical tool - in other words it's great for demos and communicating both the input and output of Python snippets (e.g. `nbviewer <http://nbviewer.ipython.org/>`_.

Many of the full-size examples of UrbanSim on this site are presented in notebooks.  The notebook is a fantastic tool to develop snippets of code a few lines at a time, and to capture and communicate higher-level workflows.  We use notebooks a LOT, and they are a wonderful way to avoid the command line in a cross-platform way.

In many cases, you can write entire UrbanSim models in the notebook, but this is not generally considered the best practice.  It's entirely up to you, though we are more than happy to share with you our insights from many hours of use.

A Gentle Introduction to UrbanSim
---------------------------------

Background
~~~~~~~~~~

UrbanSim has been an active research project since the late 1990's, and has undergone continual re-thinking, and re-engineering over the ensuing years, as documented in many of the `accumulated research papers <http://urbansim.org/Research/ResearchPapers>`_.  Below is a brief, high-level summary of UrbanSim in only a few paragraphs from a modeling/programmer perspective.  In pseudocode, UrbanSim can be boiled down to a series of models estimated and then simulated in sequence.::

    for model in models:
        model.estimate(model_configuration_parameters)
    for i in range(NUMYEARSINSIMULATION):
        for model in models:
            model.simulate(model_configuration_parameters)

The set of models varies among the many UrbanSim applications to different regions, due to the data availability and cleanliness, time and resources that could be devoted to the project, and specific research questions that motivated the projects.  The set of models almost always includes at least the following:

Residential Real Estate Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Hedonic Regression Models** estimate and predict real estate prices for different residential building types

* **Location Choice Models** estimate and predict where different types of households will choose to live, usually segmented by income and sometimes by other demographics.  These models are generally coupled with relocation models to capture the varying rates of relocation by households of different demographics.

* **Transition models** generate new households/persons to match *control totals* that specify the growth of households by demographics makeup.

Non-residential Real Estate Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Hedonic Regression Models** are analogous to the above except for modeling the rent received on non-residential building types.

* **Location Choices Models** are analagous to the above except for modeling the location choices of jobs/establishments, and are usually segmented by employment sector (also includes relocation rates)

* **Transition models** generate new establishments/firms to match *control totals* that specify the growth of businesses usually by sector.

Real Estate Development Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some representation of real estate development must be captured to accurately represent regional real estate markets.  In UrbanSim there are several options for modeling the development process, but most users are now moving to the Pro Forma based modeling approach.

* **Development Project Location Choice Models** are the easiest way to represent development, which  sample from all recent development projects, estimate a model on where development is currently being located, and find an appropriate location for a copied development.

* **Pro Forma Developer Models** take the perspective of the developer and measure the profitability of a proposed development by predicting the cash flows from the predicted rent or sales price in a given submarket and comparing these inflows to the anticipated development costs of the project.

  Development will only happen where the predicted rent is high enough to cover costs of construction and a moderate profit, and will occur roughly to meet demand based on the location choice models and control totals.

  This type of developer model is highly flexible and can account for various planning policies including affordable housing, parking requirements, subsidies of various kinds, density bonuses, and other similar policies.

  Development regulations such as comprehensive plans and zoning provide regulatory constraints on what types of developments and what densities can be considered by the model.

It should be noted that many other kinds of models can be included in the simulation loop as well.  Travel model outputs are included where possible as they provide many right-hand-side variables to the models described above.

In general, any Python script that reads and writes data can be included to help answer a specific research question or to model a certain real-world behavior - models can even be parameterized in JSON or YAML and included in the standard model set and an ever-increasing set of functionality will be added over time.