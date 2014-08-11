Tutorials
=========

Basic Example - Residential Price Hedonic
-----------------------------------------


These components are described in more detail in the link above, but an example of how they tie together can be described here. As the canonical case, take the example of a residential sales hedonic model used to perform an ordinary least squares regression on a table of building data. The best practice would be to store the building data in an Pandas HDFStore, and can include millions of rows (all of the buildings in a region) and attributes like square footage, lot size, number of bedrooms and bathrooms and the like. Importantly, the dependent variable should also be included which in this case might be the assessed or observed price of each unit.

Now, a typical setup would pass the buildings dataframe to a Buildings "view" which is defined in dataset.py like the one here. The view is then accessed in a models.py file like it is here. Finally, a model entry point is defined which combines the view with a model configuration file which is done here. The model configuration file specifies the small number of parameters necessary to build the model, most notably the actual specification of dependent and independent variables, which is done with R-like syntax using patsy.

A process like the one described above is then repeated for each model. Note that there is often some overlap in data needs for different models - for instance all three hedonic price models in this implementation use the same buildings view to compute the relevant variables (although the variables that are utilized are often different). This is why they can be thought of as separate modules in which dataset.py provides views of all the basic objects used by UrbanSim and models.py creates model entry points which combine the relevant views with configuration files (and occasionally custom code) to capture the behavior of interest to the urban modeler.

Complete Example - San Francisco UrbanSim Structure
---------------------------------------------------

A complete example of the latest UrbanSim framework is now being maintained on `GitHub <https://github.com/synthicity/sanfran_urbansim>`_.  The example requires that the UrbanSim package is already installed (no other dependencies are required).  The example is maintained under Travis Continuous Integration so should always run with the latest version of UrbanSim.

The example has a number of Python modules including ``dataset.py``, ``models.py``, ``assumptions.py``, and ``variables.py`` which will be discussed one at a time below.  The modules are then used in *workflows* which are IPython Notebooks and will be described in detail in the next section.

Tables
~~~~~~


Assumptions
~~~~~~~~~~~

Variables
~~~~~~~~~

Models
~~~~~~

Model Configuration
~~~~~~~~~~~~~~~~~~~

Complete Example - San Francisco UrbanSim Workflows
---------------------------------------------------

Estimation Workflow
~~~~~~~~~~~~~~~~~~~

Simulation Workflow
~~~~~~~~~~~~~~~~~~~

Exploration Workflow
~~~~~~~~~~~~~~~~~~~~

Specifying Scenario Inputs
--------------------------

Control Totals
~~~~~~~~~~~~~~

Zoning Changes
~~~~~~~~~~~~~~

Fees and Subsidies
~~~~~~~~~~~~~~~~~~

Model Implementation Choices
----------------------------

UrbanAccess or Zones
~~~~~~~~~~~~~~~~~~~~

Geographic Detail
~~~~~~~~~~~~~~~~~

Configuration of Models
~~~~~~~~~~~~~~~~~~~~~~~

Dealing with NaNs
~~~~~~~~~~~~~~~~~



