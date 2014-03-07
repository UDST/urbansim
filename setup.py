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
                 'License :: OSI Approved :: GNU Affero General Public License v3'],
    packages=['synthicity'],
    package_data={'': ['*.py',
                       'urbansim/*.py',
                       'urbansimd/*.py',
                       'utils/*.py']}
)
