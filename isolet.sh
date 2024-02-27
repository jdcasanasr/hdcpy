#!/bin/bash
# Loop through input arguments
for i in {1..3}
    do
        rm -rf __pycache__
        python3 isolet_unified_test.py 2000 9
    done