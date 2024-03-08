#!/bin/bash
for i in {20..100..10}
do
    for j in {1000..10000..1000}
    do
        echo "HDCpy | ISOLET | Number of Dimensions = $j | Number of Quantization Levels = $i"
        for k in {1..3}
        do
            rm -rf __pycache__
            accuracy=$(python3 isolet_unified_test.py "$j" "$i")
            echo "$j, $i, $accuracy" >> isolet.log
        done
    done
done