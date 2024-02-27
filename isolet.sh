#!/bin/bash
for i in {2..10}
do
    for j in {1000..10000..1000}
    do
        echo "Number of Dimensions = $j | Number of Quantization Levels = $i"
        for k in {1..3}
        do
            rm -rf __pycache__
            python3 isolet_unified_test.py "$j" "$i"
        done
    done
done