#!/bin/bash

#SBATCH --output=slurm_test__%j.out  
#SBATCH -p 3gpuq  

OUTPUTDIM=`am-info exp/tri5c_multi_ali_si84/final.mdl | grep 'pdfs' | awk '{print $4}'`
ls exp/tri5c_multi_ali_si84
ls py_src/
echo "$PWD"
echo $OUTPUTDIM
echo "$OUTPUTDIM"
