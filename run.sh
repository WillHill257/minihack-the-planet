#!/bin/bash
# file to manage running of slurm scripts

# expect 2 inputs
if [ $# != 1 ]
then
    echo "Usage: $0 <partition>"
    echo "Example: $0 batch"
    exit
fi

# run the file
sbatch --partition=$1 --nodes=1 --job-name=minihack --output=cluster_output.log --error=cluster_error.log slurm.sh

