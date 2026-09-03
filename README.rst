UrbanSim
========

.. image:: https://img.shields.io/pypi/v/urbansim.svg
    :target: https://pypi.python.org/pypi/urbansim/
    :alt: Latest Version

.. image:: https://github.com/UDST/urbansim/actions/workflows/test.yml/badge.svg?branch=main
   :alt: Build Status
   :target: https://github.com/UDST/urbansim/actions/workflows/test.yml

UrbanSim is a platform for building statistical models of cities and regions. These models help forecast long-range patterns in real estate development, demographics, and related outcomes, under various policy scenarios.

This ``urbansim`` Python library is a core component. It contains tools for statistical estimation and simulation; domain-specific logic about housing markets, household relocation, and other processes; and frameworks and utilities for assembling a model. 

Project scope
-------------

**Status:** Active

**Mission:** The UrbanSim Python library provides methods and reusable model
components for building self-managed simulations of urban development,
household and employment location, real estate markets, and related regional
change.

**Architecture:** UrbanSim is a portable, self-managed Python library designed
primarily for conventional CPU-based execution. It provides reusable model
components and interfaces that can also be implemented by other execution
engines.

The project maintains and develops:

* statistical model components used in urban simulation;
* location-choice, relocation, transition, and development models;
* real-estate feasibility and related urban-development methods;
* estimation and simulation utilities;
* model APIs and configuration patterns; and
* reusable components for assembling regional UrbanSim implementations.

UrbanSim is designed to work with other UDST libraries and with external data,
estimation, accessibility, and workflow systems through documented Python
interfaces and model specifications.

Development of urban-simulation methods and reusable model components is
welcome within this mission and architecture. Material changes to the
project's mission or execution architecture are considered through UDST's
organization-level governance process.

See the `UDST Project Directory
<https://github.com/UDST/.github/blob/main/PROJECTS.md>`__ and
`Open-source projects and commercial offerings
<https://github.com/UDST/.github/blob/main/OPEN_SOURCE_AND_COMMERCIAL.md>`__
for organization-wide project status and policy.

How it works
------------

Operational UrbanSim models begin with detailed data about a particular region, and then estimate and validate a system of interconnected model components. Full models draw on a number of libraries: not just ``urbansim``, but also `Orca <https://github.com/UDST/orca>`__ for task orchestration, `Synthpop <https://github.com/UDST/synthpop>`__ for population synthesis, `Pandana <https://github.com/UDST/pandana>`__ for network analysis, and so on. Collectively, these make up the `Urban Data Science Toolkit <https://github.com/UDST>`__ (UDST).

UrbanSim models are used by public agencies, consultancies, and researchers in dozens of cities around the U.S. and world. The core platform is open source, but many operational models make use of additional cloud-hosted model building and visualization tools provided by `UrbanSim Inc. <https://urbansim.com>`__

Learn More
----------

* `An Introduction to UrbanSim <https://udst.github.io/urbansim/gettingstarted.html#a-gentle-introduction-to-urbansim>`__

* `UrbanSim for San Francisco: An example implementation <https://github.com/UDST/sanfran_urbansim>`__

* `UrbanSim Inc. <https://urbansim.com>`__

Installation
------------

* ``pip install urbansim``

* ``conda install urbansim --channel conda-forge``

Technical documentation
-----------------------

* `Getting started <https://udst.github.io/urbansim/gettingstarted.html>`__

* `Full documentation <https://udst.github.io/urbansim/>`__

* Other `UDST <https://docs.udst.org>`__ libraries

* Documentation for `UrbanCanvas <https://cloud.urbansim.com/docs/>`__, the UrbanSim cloud platform
