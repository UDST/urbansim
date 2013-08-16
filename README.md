urbansim
========

New lightweight version of UrbanSim, a tool for forecasting future growth in real estate markets

![synthicity](http://www.synthicity.com/uploads/1/8/3/2/18327643/9164254_orig.png)

Concept
-------

This new project is an experimental branch aimed at reducing the learning curve of using the UrbanSim methodology.  From the ground up, it is a tool to make creating that first urban model as trivial as any other Python package available today.  Soon we will contribute the package to the Python Package Index and you will be able to "easy\_install urbansim".

This UrbanSim package uses no code from the svn repository currently hosted on urbansim.org, though it makes every effort to implement the exact same methodology.  We lean heavily on the PyData [1] community to make our work easier - Pandas, HDF5 file storage, and statsmodels are ubiquitous in this work.  These Python libraries essentially replace the UrbanSim Dataset class, tools to read and write from other storage, and some of the statistical estimation currently present in UrbanSim.  

This makes our task easier as we can focus on urban modeling and leave the infrastructure to the wider Python community.  The Pandas [2] library is the vernacular of the new UrbanSim, which is an extremely well documented library which has thousands of questions asked and answered on the internet as well as a printed book [3].  As such, we won't discuss Pandas much here - it assumed you can find that information elsewhere.

Currently, this framework implements a set of JSON requests to describe urban models.  JSON [4] is a very small lanaguage designed to communicate requests from client to server and back via the internet.  Although we don't currently implement a web service for estimating models, this is a clear pathway for the future.  For now JSON files are simply the API by which a user can configure an urban model, and the currently supported models are enough to create a basic 30 year simulation that is the prototypical use case for UrbanSim.  These models are 1) hedonic price models 2) MNL and nested location choice models 3) transition models for increasing populations 4) and relocation rate models to specify the rates of the population that moves each simulation year.

On the other hand, if you're a person who just wants to create the easiest and most straightforward location choice model around and map it, you've also come to the right place.

[1] (http://pydata.org)
[2] (http://pandas.pydata.org)
[3] (http://www.amazon.com/Python-Data-Analysis-Wes-McKinney/dp/1449319793/ref=sr_1_1?ie=UTF8&qid=1376679332&sr=8-1&keywords=python+pandas)
[4] (http://en.wikipedia.org/wiki/JSON)`

Installation
---------------

Download and install Anaconda Python distribution:

http://www.continuum.io/downloads

Get repository:

```
>> git clone https://github.com/fscottfoti/urbansim.git
```

Set environment variables:

```
>> cd urbansim
>> export PYTHONPATH=$PWD
>> export DATA_HOME=$PWD
```

Download data:

```
>> curl -k -o data/mrcog.zip https://dl.dropboxusercontent.com/u/2815546/mrcog.zip
>> unzip -d data data/mrcog.zip
```

Run:

```
>> cd example
>> ./run.sh
```
