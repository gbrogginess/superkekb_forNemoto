#!/bin/bash
source /afs/cern.ch/user/g/gbroggi/work/private/collimation_environment/activate_environment.sh

PYTHON_SCRIPT=/afs/cern.ch/user/g/gbroggi/work/private/2026/superkekb_forNemoto/001_track_injection/001_track_injection.py

python $PYTHON_SCRIPT $1
