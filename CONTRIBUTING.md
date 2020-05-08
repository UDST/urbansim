Thanks for using UrbanSim! 

This is an open source project that's part of the Urban Data Science Toolkit. Development and maintenance is a collaboration between UrbanSim Inc, U.C. Berkeley's Urban Analytics Lab, and other contributors.

You can contact Sam Maurer, the lead maintainer, at `maurer@urbansim.com`.


## If you have a problem:

- Take a look at the [open issues](https://github.com/UDST/urbansim/issues) and [closed issues](https://github.com/UDST/urbansim/issues?q=is%3Aissue+is%3Aclosed) to see if there's already a related discussion

- Open a new issue describing the problem -- if possible, include any error messages, the operating system and version of python you're using, and versions of any libraries that may be relevant


## Feature proposals:

- Take a look at the [open issues](https://github.com/UDST/urbansim/issues) and [closed issues](https://github.com/UDST/urbansim/issues?q=is%3Aissue+is%3Aclosed) to see if there's already a related discussion

- Post your proposal as a new issue, so we can discuss it (some proposals may not be a good fit for the project)

- Please note that active development of certain UrbanSim components has moved to stand-alone libraries in UDST: Developer, Choicemodels, UrbanSim Templates


## Contributing code:

- Create a new branch of `UDST/urbansim`, or fork the repository to your own account

- Make your changes, following the existing styles for code and inline documentation

- Add [tests](https://github.com/UDST/urbansim/tree/master/urbansim/tests) if possible!

- Open a pull request to the `UDST/urbansim` dev branch, including a writeup of your changes -- take a look at some of the closed PR's for examples

- Current maintainers will review the code, suggest changes, and hopefully merge it!


## Updating the documentation: 

- See instructions in `docs/README.md`


## Preparing a release:

- Make a new branch for release prep

- Update the version number and changelog
  - `CHANGELOG.md`
  - `setup.py`
  - `urbansim/__init__.py`
  - `docs/source/index.rst`

- Make sure all the tests are passing, and check if updates are needed to `README.md` or to the documentation

- Open a pull request to the master branch to finalize it

- After merging, tag the release on GitHub and follow the distribution procedures below


## Distributing a release on PyPI (for pip installation):

- Register an account at https://pypi.org, ask one of the current maintainers to add you to the project, and `pip install twine`

- Check out the copy of the code you'd like to release

- Run `python setup.py sdist bdist_wheel --universal`

- This should create a `dist` directory containing two package files -- delete any old ones before the next step

- Run `twine upload dist/*` -- this will prompt you for your pypi.org credentials

- Check https://pypi.org/project/urbansim/ for the new version


## Distributing a release on Conda Forge (for conda installation):

- The [conda-forge/urbansim-feedstock](https://github.com/conda-forge/urbansim-feedstock) repository controls the Conda Forge release

- Conda Forge bots usually detect new releases on PyPI and set in motion the appropriate feedstock updates, which a current maintainer will need to approve and merge

- Check https://anaconda.org/conda-forge/urbansim for the new version (may take a few minutes for it to appear)
