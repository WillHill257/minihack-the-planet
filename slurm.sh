#!/bin/bash

# run the script
cd $SLURM_SUBMIT_DIR # should be .../cryptic-crossword-rationale/

# activate python virtual environment
source env/bin/activate

python3 main.py

# deactivate python virtual environment
deactivate

