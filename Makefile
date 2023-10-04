# Based on https://github.com/biocore/empress/blob/master/Makefile

all: stylecheck pytest documentation

stylecheck:
	flake8 birdman/*.py tests/*.py setup.py

pytest:
	pytest tests

documentation:
	cd docs && make html
