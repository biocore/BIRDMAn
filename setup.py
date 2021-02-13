from setuptools import find_packages, setup

__version__ = "0.0.1"

setup(
    name="birdman",
    author="Gibraan Rahman",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["*.stan"]},
    install_requires=[
        "cmdstanpy",
        "dask",
        "biom-format",
        "patsy",
        "xarray"
    ]
)
