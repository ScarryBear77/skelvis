#!/bin/bash

rm -rf build/
rm -rf dist/
rm -rf skelvis.egg-info/

python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
