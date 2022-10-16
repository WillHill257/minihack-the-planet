#!/bin/bash
# file to manage running of slurm scripts

# expect 2 inputs
if [ $# != 2 ]
then
    echo "Usage: $0 <partition> <experiment location>"
    echo "Example: $0 batch experiments/one"
    exit
fi

# check if models folder exists
if [ ! -d "./models" ]
then
    mkdir ./models
fi 

# make the experiment folder
mkdir -p $2

# run the file
sbatch --partition=$1 --nodes=1 --job-name=minihack --output=$2/cluster_output.log --error=$2/cluster_error.log slurm.sh

