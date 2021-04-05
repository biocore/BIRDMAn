from setuptools import find_packages, setup

__version__ = "0.0.1"

# Inspired by the EMPress classifiers
classifiers = """
    Development Status :: 2 - Pre-Alpha
    License :: OSI Approved :: BSD License
    Programming Language :: Python :: 3.7
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3.9
    Intended Audience :: Science/Research
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
    Topic :: Scientific/Engineering :: Bio-Informatics
"""
short_desc = "Framework for differential abundance using Bayesian inference"

setup(
    name="birdman",
    author="Gibraan Rahman",
    author_email="grahman@eng.ucsd.edu",
    description=short_desc,
    url="https://github.com/gibsramen/BIRDMAn",
    version=__version__,
    license='BSD-3-Clause',
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["*.stan"]},
    install_requires=[
        "numpy",
        "cmdstanpy",
        "dask[complete]",
        "biom-format",
        "patsy",
        "xarray",
        "pandas",
        "arviz"
    ],
    extras_require={"dev": ["pytest", "scikit-bio", "sphinx"]},
    classifiers=classifiers
)
