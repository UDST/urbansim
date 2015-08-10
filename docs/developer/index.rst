Real Estate Development Models
==============================

The real estate development models included in this module are designed to
implement pencil out pro formas, which generally measure the cash inflows and
outflows of a potential investment (in this case, real estate development)
with the outcome being some measure of profitability or return on investment.
Pro formas would normally be performed in a spreadsheet program (e.g. Excel),
but are implemented in vectorized Python implementations so that many (think
millions) of pro formas can be performed at a time.

The functionality is split into two modules - the square foot pro forma and
the developer model - as there are many use cases that call for the pro formas
without the developer model.  The ``sqftproforma`` module computes real
estate feasibility for a set of parcels dependent on allowed uses, prices,
and building costs, but does not actually *build* anything (both figuratively
and literally).  The ``developer model`` decides how much to build,
then picks among the set of feasible buildings attempting to meet demand,
and adds the new buildings to the set of current buildings.  Thus
``developer model`` is primarily useful in the context of an urban forecast.

An example of the sample code required to generate the set of feasible
buildings is shown below.  This code comes from the ``utils`` module of the
current `sanfran_urbansim <https://github
.com/udst/sanfran_urbansim>`_ demo.  Notice that the SqFtProForma is
first initialized and a DataFrame of parcels is tested for feasibliity (each
individual parcel is tested for feasibility).  Each *use* (e.g. retail, office,
residential, etc) is assigned a price per parcel, typically from empirical data
of currents rents and prices in the city but can be the result of forecast
rents and prices as well.  The ``lookup`` function is then called with a
specific building ``form`` and the pro forma returns whether that form is
profitable for each parcel.

A large number of assumptions enter in to the computation of profitability
and these are set in the `SqFtProFormaConfig <#urbansim.developer.sqftproforma.SqFtProFormaConfig>`_ module, and include such things
as the set of ``uses`` to model, the mix of ``uses`` into ``forms``,
the impact of parking requirements, parking costs,
building costs at different heights (taller buildings typically requiring
more expensive construction methods), the profit ratio required,
the building efficiency, parcel coverage, and cap rate to name a few.  See
the API documentation for the complete list and detailed descriptions.

Note that unit mixes don't typically enter in to the square foot pro forma
(hence the name).  After discussions with numerous real estate developers,
we found that most developers thought first and foremost in terms of price and
cost per square foot and the arbitrage between, and second in terms of the
translation to unit sizes and mixes in a given market (also larger and
smaller units of a given unit type will typically lower and raise their
prices as stands to reason).  Since getting data on unit mixes in the current
building stock is extremely difficult, most feasibility computations here
happen on a square foot basis and the ``developer`` model below handles the
translation to units. ::

    pf = sqftproforma.SqFtProForma()

    df = parcels.to_frame()

    # add prices for each use
    for use in pf.config.uses:
        df[use] = parcel_price_callback(use)

    # convert from cost to yearly rent
    if residential_to_yearly:
        df["residential"] *= pf.config.cap_rate

    d = {}
    for form in pf.config.forms:
        print "Computing feasibility for form %s" % form
        d[form] = pf.lookup(form, df[parcel_use_allowed_callback(form)])

    far_predictions = pd.concat(d.values(), keys=d.keys(), axis=1)

    sim.add_table("feasibility", far_predictions)


The ``developer model`` is responsible for picking among feasible buildings
in order to meet demand.  An example usage of the model is shown below - which
is also lifted form the `sanfran_urbansim <https://github.com/udst/sanfran_urbansim>`_ demo.

This module provides a simple utility to compute the number of units (or
amount of floorspace) to build.  Although the vacancy rate *can* be applied
at the regional level, it can also be used to meet vacancy rates at a
sub-regional level.  The developer model itself is agnostic to which parcels
the user passes it, and the user is responsible for knowing at which level of
geography demand is assumed to operate.  The developer model then chooses
which buildings to "build," usually as a random choice weighted by profitability.
This means more profitable buildings are more likely to be built although
the results are a bit stochastic.

The only remaining steps are then "bookkeeping" in the sense that some
additional fields might need to be added (``year_built`` or a conversion from
developer ``forms`` to ``building_type_ids``).  Finally the new buildings
and old buildings need to be merged in such a way that the old ids are
preserved and not duplicated (new ids are assigned at the max of the old
ids+1 and then incremented from there).  ::

    dev = developer.Developer(feasibility.to_frame())

    target_units = dev.\
        compute_units_to_build(len(agents),
                               buildings[supply_fname].sum(),
                               target_vacancy)

    new_buildings = dev.pick(forms,
                             target_units,
                             parcel_size,
                             ave_unit_size,
                             total_units,
                             max_parcel_size=max_parcel_size,
                             drop_after_build=True,
                             residential=residential,
                             bldg_sqft_per_job=bldg_sqft_per_job)

    if year is not None:
        new_buildings["year_built"] = year

    if form_to_btype_callback is not None:
        new_buildings["building_type_id"] = new_buildings["form"].\
            apply(form_to_btype_callback)

    all_buildings = dev.merge(buildings.to_frame(buildings.local_columns),
                              new_buildings[buildings.local_columns])

    sim.add_table("buildings", all_buildings)

.. toctree::
   :maxdepth: 2


Square Foot Pro Forma API
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: urbansim.developer.sqftproforma
   :members:

Developer Model API
~~~~~~~~~~~~~~~~~~~

.. automodule:: urbansim.developer.developer
   :members:
