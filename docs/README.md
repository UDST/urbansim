This folder generates the UrbanSim online documentation, hosted at https://udst.github.io/urbansim/.

### How it works

HTML files are generated using [Sphinx](http://sphinx-doc.org) and hosted with GitHub Pages from the `gh-pages` branch of the repository. The online documentation is rendered and updated **manually**. 

### Editing the documentation

The files in `docs/source`, along with docstrings in the source code, determine what appears in the rendered documentation. Here's a [good tutorial](https://pythonhosted.org/an_example_pypi_project/sphinx.html) for Sphinx.

### Previewing changes locally

Install the copy of UrbanSim that the documentation is meant to reflect. Install the documentation tools.

```
pip install . 
pip install sphinx sphinx_rtd_theme numpydoc
```

Build the documentation. There should be status messages and warnings, but no errors.

```
cd docs
sphinx-build -b html source build
```

The HTML files will show up in `docs/build/`. 

### Uploading changes

Clone a second copy of the repository and check out the `gh-pages` branch. Copy over the updated HTML files, commit them, and push the changes to GitHub.
