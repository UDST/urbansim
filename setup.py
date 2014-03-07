from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup

setup(
    name='urbansim',
    version='0.2dev',
    description='Tool for modeling metropolitan real estate markets',
    author='Synthicity',
    author_email='ffoti@berkeley.edu',
    license='AGPL',
    url='https://github.com/synthicity/urbansim',
    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 2.7',
                 ('License :: OSI Approved :: '
                  'GNU Affero General Public License v3')],
    packages=['synthicity'],
    package_data={'': ['*.py',
                       'urbansim/*.py',
                       'urbansimd/*.py',
                       'utils/*.py']},
    install_requires=['Django>=1.6.2',
                      'jinja2>=2.7.2',
                      'numpy>=1.8.0',
                      'pandas>=0.13.1',
                      'patsy>=0.2.1',
                      'pytables>=3.1.0',
                      'scipy>=0.13.3',
                      'shapely>=1.3.0',
                      'simplejson>=3.3.3',
                      'statsmodels>=0.5.0']
)
