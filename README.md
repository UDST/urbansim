urbansim
========

[![build status](http://img.shields.io/travis/synthicity/urbansim.svg)](https://travis-ci.org/synthicity/urbansim) [![test coverage](http://img.shields.io/coveralls/synthicity/urbansim.svg)](https://coveralls.io/r/synthicity/urbansim)

New lightweight version of UrbanSim, a tool for modeling metropolitan real estate markets

![synthicity](http://www.synthicity.com/uploads/1/8/3/2/18327643/9164254_orig.png)

Documentation will be kept on the github wiki: https://github.com/synthicity/urbansim/wiki

Concept
-------

This new code base is a streamlined complete re-implementation of the longstanding UrbanSim project [1] aimed at *reducing the complexity* of using the UrbanSim methodology.  Redesigned from the ground up, it is a tool to make installation as trivial as any other Python package available today, and making modeling much more widely accessible to planners and new modelers.

This UrbanSim package uses no code from the svn repository currently hosted on urbansim.org, though it makes every effort to implement the exact same methodology.  We lean heavily on the PyData [2] community to make our work easier - *Pandas, HDF5 file storage, and statsmodels* are ubiquitous in this work.  These Python libraries essentially replace the UrbanSim Dataset class, tools to read and write from other storage, and some of the statistical estimation currently present in UrbanSim.

This makes our task easier as we can focus on urban modeling and leave the infrastructure to the wider Python community.  The Pandas [3] library is the vernacular of the new UrbanSim, which is an extremely well documented library that has thousands of questions asked and answered on the internets as well as a *printed book* [4].  As such, we won't discuss Pandas much here - it is assumed you can find that information elsewhere.

Currently, this framework implements a set of JSON requests to describe urban models.  JSON [5] is a very basic lanaguage designed to communicate requests from client to server and back on the internets.  Although we don't currently implement a web service for estimating models, this is a clear pathway for the future.  Put simply, *JSON files are the API* by which a user can configure an urban model (though we certainly don't forbid getting your hands dirty and writing a model in Python).

Currently supported models are enough to create a basic 30 year simulation that is the prototypical use case for UrbanSim.  These models are 1) *hedonic price models* 2) MNL and nested *location choice models* 3) transition models for increasing population, jobs, and development and 4) relocation rate models to specify the movement of population and jobs.  We have also implemented a new real estate development model using proforma analysis.

**If you just want to begin experimenting and creating regression and choice models and map their inputs and outputs, you've also come to the right place.**

[1] http://urbansim.org/

[2] http://pydata.org

[3] http://pandas.pydata.org

[4] http://www.amazon.com/Python-Data-Analysis-Wes-McKinney/dp/1449319793

[5] http://en.wikipedia.org/wiki/JSON

Installation Instructions for users who don't want to mess with GitHub (most people)
---------------
We are working on it.  As soon as we are ready, we will create an installer that will make installation trivial for end-users.
We do strongly recommend that you contact info@synthicity.org about your project to make sure you can get professional support when you need it, and know what you are getting into.  For major applied projects, professional support is highly recommended.

Installation Instructions for programmers
---------------

Download and install Anaconda Python distribution (early reviews on the v1.8 Anaconda installer are that there's something wrong with Pandas - not sure what the specifiecs are yet, but we suggest v1.6 of the installer with Pandas v0.11 - upgrading will cause Pandas to crash on many operations!):

http://www.continuum.io/downloads

Get repository:

```
>> git clone https://github.com/synthicity/urbansim.git
```

Set environment variables:

```
>> cd urbansim
>> export PYTHONPATH=$PWD
>> export DATA_HOME=$PWD
```

!!! Contact info@synthicity.com to get sample dataset - this example won't run without it. !!!
<!--Download data:

```
>> curl -k -o data/mrcog.zip https://dl.dropboxusercontent.com/u/2815546/mrcog.zip
>> unzip -d data data/mrcog.zip
```-->

Run:

```
>> cd example
>> ./run.sh
```
