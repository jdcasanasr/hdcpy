#!/bin/bash
for i in {1..2}
    do
        echo "Iteration: $i"

        rm -rf __pycache__
        accuracy=$(python3 isolet_unified_test.py "$1" "$2")
        echo "$1, $2, $accuracy" >> isolet.log
    done