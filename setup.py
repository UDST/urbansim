# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='urbansim',
    version='0.1.0',
    description='Updated version of UrbanSim based on Pandas',
    author='Fletcher Foti',
    author_email='ffoti@berkeley.edu',
    license='AGPL',
    url='https://github.com/fscottfoti/urbansim',
    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 2.7',
                 'License :: OSI Approved :: AGPL License'],
    packages=['synthicity'],
    package_data={'': ['*.py',
                       'urbansim/*.py',
                       'urbansimd/*.py',
                       'utils/*.py']}
)
