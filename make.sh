#!/usr/bin/env bash

cd utils/pyvotkit
python2 setup.py build_ext --inplace
cd ../../

cd utils/pysot/utils/
python2 setup.py build_ext --inplace
cd ../../../
