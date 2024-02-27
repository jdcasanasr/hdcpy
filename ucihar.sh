#!/bin/bash
# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <argument1> <argument2>"
    exit 1
fi

# Assign command-line arguments to variables
arg1=$1
arg2=$2

# Loop through input arguments
for i in {1..3}
do
    rm -rf __pycache__
    python3 ucihar_unified_test.py "$arg1" "$arg2"
done