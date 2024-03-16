#!/bin/bash

total_iterations=$(( (100 / 10) * (10000 / 1000) * 5 ))
progress=0
python_script_name=$(basename "$1")

if [ ! -d "../logs" ]; then
    mkdir "../logs"
fi

# Check if a previous log exists, and erase it.
if [ -f "../logs/${python_script_name//.py}.csv" ]; then
    rm "../logs/${python_script_name//.py}.csv"
fi

if [ -d "../__pycache" ]; then
    # For some reason, __pycache__ keeps popping up!
    rm -rf __pycache__
fi

# Loop for benchmarking
for number_of_quantization_levels in {10..100..10}
do
    for number_of_dimensions in {1000..10000..1000}
    do
        for number_of_tests in {1..5}
        do
            python3 "$1" "$number_of_dimensions" "$number_of_quantization_levels" >> "../logs/${python_script_name//.py}.csv"

            # Output progress to the user.
            ((progress++))
            progress_percentage=$(( (progress * 100) / total_iterations ))
            echo -ne "Benchmarking: $python_script_name | Progress: $progress_percentage%\r"
        done
    done
done

echo "Benchmark Complete!"