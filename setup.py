# Inspired by the EMPress setup.py file
import ast
import re
from setuptools import find_packages, setup

# version parsing from __init__ pulled from Flask's setup.py
# https://github.com/mitsuhiko/flask/blob/master/setup.py
_version_re = re.compile(r"__version__\s+=\s+(.*)")

with open("birdman/__init__.py", "rb") as f:
    hit = _version_re.search(f.read().decode("utf-8")).group(1)
    version = str(ast.literal_eval(hit))


classifiers = """
    Development Status :: 3 - Alpha
    License :: OSI Approved :: BSD License
    Programming Language :: Python :: 3.7
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3.9
    Intended Audience :: Science/Research
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
    Topic :: Scientific/Engineering :: Bio-Informatics
"""
short_desc = (
    "Framework for differential microbiome abundance using Bayesian "
    "inference"
)

with open("README.md") as f:
    long_description = f.read()

setup(
    name="birdman",
    author="Gibraan Rahman",
    author_email="grahman@eng.ucsd.edu",
    description=short_desc,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gibsramen/BIRDMAn",
    version=version,
    license="BSD-3-Clause",
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["*.stan"]},
    install_requires=[
        "numpy",
        "cmdstanpy>=1.0.1",
        "scipy",
        "biom-format",
        "patsy",
        "xarray",
        "pandas",
        "arviz>=0.11.3",
        "matplotlib"
    ],
    extras_require={"dev": ["pytest", "scikit-bio", "sphinx"]},
    classifiers=[s.strip() for s in classifiers.split('\n') if s]
)
