#!/bin/bash
source /afs/cern.ch/user/g/gbroggi/work/private/collimation_environment/activate_environment.sh

PYTHON_SCRIPT=/afs/cern.ch/user/g/gbroggi/work/private/2026/superkekb_forNemoto/003_fma_study/003_fma_study.py

python $PYTHON_SCRIPT $1
