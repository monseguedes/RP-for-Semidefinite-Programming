#!/bin/sh
#
# A shell script to setup a virtual environment. Run this script from the top
# directory of the project:
#
# $ ./setup_python_env.sh

set -x
set -e

if [ -d "env" ]; then
    echo "virtual environment exists"
    exit 0
fi

# ~/Documents/GitHub/Virtual_Environment/KrakenflexVErn/bin/python -m venv env
/usr/bin/python3.6 -m venv env

. ./env/bin/activate

pip install -U pip
pip install \
    black \
    gurobipy \
    isort \
    matplotlib \
    networkx \
    numpy \
    Mosek \
    pandas \
    scipy \ 
    seaborn \
    yaml 
    
