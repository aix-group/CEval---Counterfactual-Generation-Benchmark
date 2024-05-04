#!/bin/bash

# List of arguments
#arguments=("results/imdb/llama/results.csv" "results/imdb/mice/results.csv" "results/imdb/crowd/results.csv" "results/imdb/expert/results.csv" "results/imdb/gbda/results.csv" "results/imdb/crest/results.csv")
arguments=("results/imdb/crowd/results.csv" "results/imdb/expert/results.csv")
# Path to the Python file
python_file="run_benchmark.py"

additional_args=("-task" "imdb" "-path_csv")

# Loop through the arguments and run the Python script
for arg in "${arguments[@]}"; do
    echo "Running with argument: $arg"
    python3 "$python_file" "${additional_args[@]}" "$arg"
done
