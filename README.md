# Benchmarking for Counterfactual Text Generation

**Setup:**
- Install necessary packages by running `pip install -r requirement.txt`.
- To run the benchmark, ensure that the generated results are available in the form of a .csv file with two columns: `orig_text` and `gen_text`.

**Counterfactual Metrics:**
- To obtain results for counterfactual metrics, execute the following command:
  ```bash
  python run_benchmark.py -task TASK -path_csv PATH_CSV
```
- `TASK` can be either "snli" or "imdb".
- `PATH_CSV` is the path to the result .csv file.

- For LLMs evaluation results, place the results in the following folder structure: `results/{method}/results.csv`.
- Add the method to the `main` function in `metrics/LLMs_evaluation.py`, then execute it using the command:
  ```bash
python metrics/LLMs_evaluation.py
```

**Note:** The generated results required to reproduce the paper are available in the `results` folder. Simply utilize them and run the script to reproduce the paper.