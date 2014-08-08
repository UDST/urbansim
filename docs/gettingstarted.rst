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
