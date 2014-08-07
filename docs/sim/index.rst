Simulation Framework
====================

Introduction
------------

UrbanSim's simulation framework allows you to register data and functions
that operate on that data. The main components of a simulation include:

* Tables

  * Most of the modeling functionality in UrbanSim works with pandas
    `DataFrames`_. Tables can be registered as plain DataFrames or as
    functions that return DataFrames.

* Columns

  * Columns can be dynamically added to tables by registering
    individual pandas `Series`_ instances or by registering functions
    that return Series.

* Models

  * Models are Python functions registered with the simulation.
    Most models will update the simulation somehow, maybe by updating
    a column in a table or adding rows to a table, but they can theoretically
    do anything.

* Injectables

  * Models may need to make use of data that are not in a table/column.
    For these you can register any object/any function as an injectable
    to make it available to models.

The framework offers some conveniences for streamlining
the simulation process:

* Dependency injection

  * When you register any function UrbanSim inspects the argument list
    and stores the argument names. Then, when the function needs to be
    evaluated, UrbanSim matches those argument names to registered tables
    and injectables, and calls the function with those things (in turn
    calling any other functions necessary and injecting other things).

* Functions as data

  * If something needs to be recomputed on-demand you can register
    a function that returns your table/column/injectable. That function
    will be evaluated anytime the variable is used in the simulation so
    that the value is always current.

* Caching

  * Have some data that needs to be computed, but not very frequently?
    You can enable caching on individual items to save time, then later clear
    the cache on just that item or clear the entire cache in one call.

* Automated Merges

  * UrbanSim can merge multiple tables to some target table once you have
    described relationships between them.

* Data archives

  * After a simulation it can be useful to look out how the data changed
    as the simulation progressed. UrbanSim can save registered tables out
    to an HDF5 file every simulation iteration or on set intervals.

.. note::
   In the documentation below the following imports are implied::

       import pandas as pd
       import urbansim.sim.simulation as sim

Tables
------

Tables are pandas `DataFrames`_.
Use the :py:func:`~urbansim.sim.simulation.add_table` function to register
a DataFrame under a given name::

    df = pd.DataFrame({'a': [1, 2, 3]})
    sim.add_table('my_table', df)

Or you can use the decorator :py:func:`~urbansim.sim.simulation.table`
to register a function that returns a DataFrame::

    @sim.table('halve_my_table')
    def halve_my_table(my_table):
        df = my_table.to_frame()
        return df / 2

By registering ``halve_my_table`` as a function its values will always be
half those in ``my_table``, even if ``my_table`` is later changed.
If you'd like a function to *not* be evaluated every time it
is used, pass the ``cache=True`` keyword when registering it.

Note that the names given to tables (and other registered things) should be
`valid Python variable names <http://en.wikibooks.org/wiki/Python_Beginner_to_Expert/Native_Types>`_
so that they can be used in dependency injection.

Here's a demo of the above table definitions shown in IPython::

    In [19]: wrapped = sim.get_table('halve_my_table')

    In [20]: wrapped.to_frame()
    Out[20]:
         a
    0  0.5
    1  1.0
    2  1.5

Table Wrappers
~~~~~~~~~~~~~~

Notice in the table function above that we had to call a
:py:meth:`~urbansim.sim.simulation.DataFrameWrapper.to_frame` method
before using the table in a math operation. The values injected into
functions are not DataFrames, but specialized wrappers.
The wrappers facilitate caching, `computed columns <#columns>`_,
and lazy evaluation of table functions. Learn more in the API documentation:

* :py:class:`~urbansim.sim.simulation.DataFrameWrapper`
* :py:class:`~urbansim.sim.simulation.TableFuncWrapper`

Table Sources
~~~~~~~~~~~~~

Sometimes you may want to set up a function that returns a DataFrame but
only have it evaluated once and thereafter have access to the DataFrame itself
as if you had used :py:func:`~urbansim.sim.simulation.add_table` to
register it. For that you can use the
:py:func:`~urbansim.sim.simulation.table_source` decorator::

    @sim.table_source('my_table')
    def my_table():
        return pd.DataFrame({'a': [1, 2, 3]})

When ``my_table`` is first injected somewhere it will be converted to a
:py:class:`~urbansim.sim.simulation.DataFrameWrapper`.

Automated Merges
~~~~~~~~~~~~~~~~

Certain analyses can be easiest when some tables are merged together,
but in other places it may be best to keep the tables separate.
UrbanSim can make these on-demand merges easy by letting you define table
relationships up front and then performing the merges for you as needed.
We call these relationships "broadcasts" (as in a rule for how to broadcast
one table onto another) and you register them using the
:py:func:`~urbansim.sim.simulation.broadcast` function.

For an example we'll first define some DataFrames that contain links
to one another and register them with the simulation::

    df_a = pd.DataFrame(
        {'a': [0, 1]},
        index=['a0', 'a1'])
    df_b = pd.DataFrame(
        {'b': [2, 3, 4, 5, 6],
         'a_id': ['a0', 'a1', 'a1', 'a0', 'a1']},
        index=['b0', 'b1', 'b2', 'b3', 'b4'])
    df_c = pd.DataFrame(
        {'c': [7, 8, 9]},
        index=['c0', 'c1', 'c2'])
    df_d = pd.DataFrame(
        {'d': [10, 11, 12, 13, 15, 16, 16, 17, 18, 19],
         'b_id': ['b2', 'b0', 'b3', 'b3', 'b1', 'b4', 'b1', 'b4', 'b3', 'b3'],
         'c_id': ['c0', 'c1', 'c1', 'c0', 'c0', 'c2', 'c1', 'c2', 'c1', 'c2']},
        index=['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9'])

    sim.add_table('a', df_a)
    sim.add_table('b', df_b)
    sim.add_table('c', df_c)
    sim.add_table('d', df_d)

The tables have data so that 'a' can be broadcast onto 'b',
and 'b' and 'c' and be broadcast onto 'd'.
We use the :py:func:`~urbansim.sim.simulation.broadcast` function
to register those relationships::

    sim.broadcast(cast='a', onto='b', cast_index=True, onto_on='a_id')
    sim.broadcast(cast='b', onto='d', cast_index=True, onto_on='b_id')
    sim.broadcast(cast='c', onto='d', cast_index=True, onto_on='c_id')

The syntax is similar to that of the
`pandas merge function <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.merge.html#pandas.merge>`_,
and indeed ``merge`` is used behind the scenes.
Once the broadcasts are defined, use the
:py:func:`~urbansim.sim.simulation.merge_tables` function to get a
merged DataFrame. Some examples in IPython::

    In [4]: sim.merge_tables(target='b', tables=[a, b])
    Out[4]:
       a_id  b  a
    b0   a0  2  0
    b3   a0  5  0
    b1   a1  3  1
    b2   a1  4  1
    b4   a1  6  1

    In [5]: sim.merge_tables(target='d', tables=[a, b, c, d])
    Out[5]:
       b_id c_id   d  c a_id  b  a
    d0   b2   c0  10  7   a1  4  1
    d3   b3   c0  13  7   a0  5  0
    d2   b3   c1  12  8   a0  5  0
    d8   b3   c1  18  8   a0  5  0
    d9   b3   c2  19  9   a0  5  0
    d4   b1   c0  15  7   a1  3  1
    d6   b1   c1  16  8   a1  3  1
    d1   b0   c1  11  8   a0  2  0
    d5   b4   c2  16  9   a1  6  1
    d7   b4   c2  17  9   a1  6  1

Note that it's the target table's index that you find in the final merged
table, though the order may have changed.
:py:func:`~urbansim.sim.simulation.merge_tables` has an optional
``columns=`` keyword that can contain column names from any the tables
going into the merge so you can limit which columns end up in the final table.
(Columns necessary for performing merges will be included whether or not
they are in the ``columns=`` list.)

.. note:: :py:func:`~urbansim.sim.simulation.merge_tables` calls
   `pandas.merge <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.merge.html#pandas.merge>`_
   with ``how='inner'``, meaning that only items that
   appear in both tables are kept in the merged table.

Columns
-------

Often, not all the columns you need for a simulation are preexisting
on your tables. You may need to collect information from other tables
or perform a calculation to generate a column. UrbanSim allows you to
register a `Series`_ or function as a column on a registered table.
Use the :py:func:`~urbansim.sim.simulation.add_column` function or
the :py:func:`_urbansim.sim.simulation.column` decorator::

    s = pd.Series(['a', 'b', 'c'])
    sim.add_column('my_table', 'my_col', s)

    @sim.column('my_table', 'my_col_x2')
    def my_col_x2(my_table):
        df = my_table.to_frame(columns=['my_col'])
        return df['my_col'] * 2

In the ``my_col_x2`` function we use the ``columns=`` keyword on
:py:meth:`~urbansim.sim.simulation.DataFrameWrapper.to_frame` to get only
the one column necessary for our calculation. This can be useful for
avoiding unnecessary computation or to avoid recursion (as would happen
in this case if we called ``to_frame()`` with no arguments).

A demonstration in IPython using the table definitions from above::

    In [29]: wrapped = sim.get_table('my_table')

    In [30]: wrapped.columns
    Out[30]: ['a', 'my_col', 'my_col_x2']

    In [31]: wrapped.local_columns
    Out[31]: ['a']

    In [32]: wrapped.to_frame()
    Out[32]:
       a my_col_x2 my_col
    0  1        aa      a
    1  2        bb      b
    2  3        cc      c

:py:class:`~urbansim.sim.simulation.DataFrameWrapper` has
:py:attr:`~urbansim.sim.simulation.DataFrameWrapper.columns`
and :py:attr:`~urbansim.sim.simulation.DataFrameWrapper.local_columns`
attributes that, respectively, list all the columns on a table and
only those columns that are part of the underlying DataFrame.

Columns are stored separate from tables so it is safe to define a column
on a table and then replace that table with something else. The column
will remain associated with the table.

Injectables
-----------

You will probably want to have things besides tables injected into functions,
for which UrbanSim has "injectables". You can register *anything* and have
it injected into functions.
Use the :py:func:`~urbansim.sim.simulation.add_injectable` function or the
:py:func:`~urbansim.sim.simulation.injectable` decorator::

    sim.add_injectable('z', 5)

    @sim.injectable('pow', autocall=False)
    def pow(x, y):
        return x ** y

    @sim.injectable('zsquared')
    def zsquared(z, pow):
        return pow(z, 2)

    @sim.table('ztable')
    def ztable(my_table, zsquared):
        df = my_table.to_frame(columns=['a'])
        return df * zsquared

Be default injectable functions are evaluated before injection and the return
value is passed into other functions. Use ``autocall=False`` to disable this
behavior and instead inject the wrapped function itself.
Like tables and columns, injectable functions can have their results
cached with ``cache=True``.

An example of the above injectables in IPython::

    In [38]: wrapped = sim.get_table('ztable')

    In [39]: wrapped.to_frame()
    Out[39]:
        a
    0  25
    1  50
    2  75

Models
------

In UrbanSim a model is a function run by the simulation framework with
dependency injection. Use the :py:func:`~urbansim.sim.simulation.model`
decorator to register a model function.
Models are important for their side-effects, their
return values are discarded. For example, a model might replace a column
in a table (a new table, though similar to ``my_table`` above)::

    df = pd.DataFrame({'a': [1, 2, 3]})
    sim.add_table('new_table')

    @sim.model('replace_col')
    def replace_col(new_table):
        new_table['a'] = [4, 5, 6]

Or update some values in a column::

    @sim.model('update_col')
    def update_col(new_table):
        s = pd.Series([99], index=[1])
        new_table.update_col_from_series('a', s)

Or add rows to a table::

    @sim.model('add_rows')
    def add_rows(new_table):
        new_rows = pd.DataFrame({'a': [100, 101]}, index=[3, 4])
        df = new_table.to_frame()
        df = pd.concat([df, new_rows])
        sim.add_table('new_table', df)

The first two of the above examples update ``my_tables``'s underlying DataFrame and
so require it to be a :py:class:`~urbansim.sim.simulation.DataFrameWrapper`.
If your table is a wrapped function, not a DataFrame, you can update
columns by replacing them entirely with a new `Series`_ using the
:py:func:`~urbansim.sim.simulation.add_column` function.

A demonstration of running the above models::

    In [68]: sim.run(['replace_col', 'update_col', 'add_rows'])
    Running model 'replace_col'
    Running model 'update_col'
    Running model 'add_rows'

    In [69]: sim.get_table('new_table').to_frame()
    Out[69]:
         a
    0    4
    1   99
    2    6
    3  100
    4  101

Though updating tables is generally how models will advance the simulation
they can do anything you like, so feel free to insert models with any
arbitrary purpose (for example, clearing cached data) into the simulation.

Running Simulations
-------------------

You start simulations by calling the :py:func:`~urbansim.sim.simulation.run`
function and listing which models you want to run.
Calling :py:func:`~urbansim.sim.simulation.run` with just a list of models,
as in the above example, will run through the models once.
To run the simulation over some years provide those years as a sequence
to :py:func:`~urbansim.sim.simulation.run`.
The variable ``year`` is provided as an injectable to model functions::

    In [77]: @sim.model('print_year')
       ....: def print_year(year):
       ....:         print '*** the year is {} ***'.format(year)
       ....:

    In [78]: sim.run(['print_year'], years=range(2010, 2015))
    Running year 2010
    Running model 'print_year'
    *** the year is 2010 ***
    Running year 2011
    Running model 'print_year'
    *** the year is 2011 ***
    Running year 2012
    Running model 'print_year'
    *** the year is 2012 ***
    Running year 2013
    Running model 'print_year'
    *** the year is 2013 ***
    Running year 2014
    Running model 'print_year'
    *** the year is 2014 ***

Archiving Data
~~~~~~~~~~~~~~

An option to the :py:func:`~urbansim.sim.simulation.run` function is to have
it save simulation data at set intervals.
Only tables are saved, as `DataFrames`_ to an HDF5 file via pandas'
`HDFStore <http://pandas.pydata.org/pandas-docs/stable/io.html#hdf5-pytables>`_
feature. If the simulation is running only one loop the tables are stored
under their registered names. If it is running multiple years the tables are
stored under names like ``'<year>/<table name>'``. For example, in the year 2020
the "buildings" table would be stored as ``'2020/buildings'``.
The ``out_interval`` keyword to :py:func:`~urbansim.sim.simulation.run`
controls how often the tables are saved out. For example, ``out_interval=5``
saves tables every fifth year. In addition, the final data is always saved
under the key ``'final/<table name>'``.

API
---

.. automodule:: urbansim.sim.simulation
   :members:

.. _DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
.. _DataFrames: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
.. _Series: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#series
