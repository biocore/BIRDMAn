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
        "numpy",
        "cmdstanpy",
        "dask[complete]",
        "biom-format",
        "patsy",
        "xarray",
        "arviz>=0.11.2"
    ],
    extras_require={"dev": ["pytest", "scikit-bio"]}
)
