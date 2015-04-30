# Install setuptools if not installed.
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup, find_packages


# read README as the long description
with open('README.rst', 'r') as f:
    long_description = f.read()

setup(
    name='urbansim',
    version='2.0.1',
    description='Tool for modeling metropolitan real estate markets',
    long_description=long_description,
    author='Synthicity',
    author_email='ffoti@synthicity.com',
    license='BSD',
    url='https://github.com/synthicity/urbansim',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: BSD License'
    ],
    package_data={
        '': ['*.html'],
    },
    packages=find_packages(exclude=['*.tests']),
    install_requires=[
        'bottle>=0.12',
        'matplotlib>=1.3.1',
        'numpy>=1.8.0',
        'pandas>=0.13.1',
        'patsy>=0.2.1',
        'prettytable>=0.7.2',
        'pyyaml>=3.10',
        'scipy>=0.13.3',
        'simplejson>=3.3',
        'statsmodels>=0.5.0',
        'tables>=3.1.0',
        'toolz>=0.7.0'
    ],
    extras_require={
        'pandana': ['pandana>=0.1']
    }
)
