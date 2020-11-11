from setuptools import find_packages, setup

setup(
    name="songbird2",
    author="Gibraan Rahman",
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["*.stan"]}
)
