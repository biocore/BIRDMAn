# Based on https://github.com/biocore/empress/blob/master/Makefile

stylecheck:
	flake8 birdman/*.py tests/*.py setup.py

pytest:
	pytest --disable-pytest-warnings
