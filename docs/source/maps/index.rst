.. _dframe-explorer:

DataFrame Explorer
==================

Introduction
------------

The DataFrame Explorer is used to create a web service within the IPython
Notebook which responds to queries from a web browser.  The REST API is
undocumented as the user does not interact with that API.  Simply call the
``start`` method below and then open `http://localhost:8765
<http://localhost:8765>`_ in any web browser.

See :ref:`exploration-workflow` for sample code from the San Francisco case study.

The dframe_explorer takes a dictionary of DataFrames which are joined to a set
of shapes for visualization.  The most common case is to use a
`geojson <http://geojson.org/>`_ format shapefile of zones to join to any
DataFrame that has a zone_id (the dframe_explorer module does the join for
you).  Then set the center and zoom level for the map, the name of the geojson
shapefile is passed, and the join keys both in the geojson file and the
DataFrames.  Below is a screenshot of the result as displayed in your web
browser.

.. image:: ../screenshots/dframe_explorer_screenshot.png

.. _dframe-explorer-website:

Website Description
-------------------

Here is what each dropdown on the web page does:

* The first dropdown gives the names of the DataFames you have passed
  ``dframe_explorer.start``
* The second dropdown allows you to choose between each of the columns in the
  DataFrame with the name from the first dropdown
* The third dropdown selects the color scheme from the
  `colorbrewer <http://colorbrewer2.org/>`_ color schemes
* The fourth dropdown sets ``quantile`` and ``equal_interval``
  `color schemes <http://www.ncgia.ucsb.edu/cctp/units/unit47/html/quanteq.html>`_
* The fifth dropdown selects the Pandas aggregation method to use
* The sixth dropdown executes the
  `.query <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html>`_
  method on the Pandas DataFrame in order to filter the input data
* The seventh dropdown executes the
  `.eval <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.eval.html>`_
  method on the Pandas DataFrame in order to create simple computed variables
  that are not already columns on the DataFrame.

What's it Doing Exactly?
------------------------

So what is this doing?  The web service is translating the drop downs to a
simple interactive Pandas statement, for example: ::

    df.groupby('zone_id')['residential_units'].sum()

The web service will print out each statement it executes.  The website then
transparently joins the output Pandas series to the shapes and create an
interactive *slippy* web map using the `Leaflet <http://leafletjs.com/>`_
Javasript library.  The code for this map is really
`quite simple <https://github.com/udst/urbansim/tree/master/urbansim/maps>`_
- feel free to browse the code and add functionality as required.

To be clear, the website is performing a Pandas aggregation on the fly.
If you have a buildings DataFrame with millions of records, Pandas will
``groupby`` the ``zone_id`` and perform an aggregation of your choice.
This is designed to give you a quickly navigable map interface to understand
the underlying disaggregate data, similar to that supplied by commercial
projects such as `Tableau <http://onlinehelp.tableau.com/current/pro/desktop/en-us/help.htm#maps.html>`_.

As a concrete example, note that the ``households`` table has a ``zone_id``
and is thus available for aggregation in ``dframe_explorer``.  Since the web
service is running aggregations on the *disaggregate* data, clicking to the
``households`` table and ``persons`` attribute and an aggregation of ``sum``
will run: ::

    households.groupby('zone_id').persons.sum()

This computes the sum of persons in each household by zone, or more simply,
the population of each zone.  If the aggregation is changed to mean, the
service will run: ::

    households.groupby('zone_id').persons.mean()

What does this compute exactly?  It computes the average number of persons per
household in each zone, or the average household size by zone.

DataFrame Explorer API
----------------------

.. automodule:: urbansim.maps.dframe_explorer
   :members: start
   :undoc-members:
