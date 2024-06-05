#!/bin/bash
arguments=(0.6)
python_file="methods/LLAMA-2/simple_prompt.py"

additional_args=("-task" "imdb" "-batch_size" 100 "-temperature")

for arg in "${arguments[@]}"; do
    echo "Running with argument: $arg"
    python3 "$python_file" "${additional_args[@]}" "$arg"
done