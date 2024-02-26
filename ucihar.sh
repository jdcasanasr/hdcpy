#!/bin/bash
# Loop through input arguments
for i in {1..3}
    do
        rm *.npy
        rm -rf __pycache__
        python3 ucihar_unified_test.py 3000 8
    done