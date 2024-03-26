#!/bin/bash
# ToDo: Make a general script, maybe parsing a .json
# or .yaml file for configuration.

#total_iterations=$(( (100 / 10) * (10000 / 1000) * 5 ))
#progress=0
python_script_name=$(basename "$1")

if [ ! -d "../logs" ]; then
    mkdir "../logs"
fi

# Check if a previous log exists, and if so, erase it.
if [ -f "../logs/${python_script_name//.py}.csv" ]; then
    rm "../logs/${python_script_name//.py}.csv"
fi

if [ -d "../__pycache" ]; then
    # For some reason, __pycache__ keeps popping up!
    rm -rf __pycache__
fi

number_of_quantization_levels=10

# Actual benchmark.

 for number_of_tests in {1..5}; do
    for number_of_dimensions in {1000..10000..1000}; do
        for _ in {1..5}; do
            # Allocate the first core to the Python interpreter.
            python3 "$1" "$number_of_dimensions" "$number_of_quantization_levels" >> "../logs/${python_script_name//.py}_same_q.csv"
            # Output progress to the user.
            ((progress++))
            #progress_percentage=$(( (progress * 100) / total_iterations ))
            #echo -ne "Benchmarking: $python_script_name | Progress: $progress_percentage%\r"
            echo -ne "Benchmarking: "$python_script_name". Please, wait.%\r"
        done
    done
done

echo "Benchmark Complete!"