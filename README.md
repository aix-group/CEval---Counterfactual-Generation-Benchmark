# CEval: A Benchmark for Evaluating Counterfactual Text Generation

This is the repository for the paper **CEval: A Benchmark for Evaluating Counterfactual Text Generation** published at INLG 2024.


**Setup:**
- Install necessary packages by running:
  ```bash
  pip install -r requirements.txt
  ```
- Ensure that the generated results are available in the form of a .csv file to run the benchmark.
  - For the IMDB task, the .csv file needs to contain 2 columns: `orig_text` and `gen_text`, corresponding to the original text and the counterfactual text. To calculate the diversity, a 3rd column called `gen_text_2` is required. An example of the .csv file can be found in `results/imdb/crowd/results.csv`.
  - For the SNLI task, the .csv file needs to contain 4 columns: `orig_premise`, `orig_hypothesis`, `gen_premise`, and `gen_hypothesis`. An example of the .csv file can be found in `results/snli/llama_02/results.csv`.

**Counterfactual Metrics:**
- To obtain results for counterfactual metrics (e.g., fliprate, distance, etc.), execute the following command:
  ```bash
  python eval_counterfactual_metrics.py -task TASK -csv_path PATH_CSV
  ```
  - `TASK`: can be either "snli" or "imdb".
  - `PATH_CSV`: the path to the result .csv file.

  - For example:
  ```bash
  python eval_counterfactual_metrics.py -task imdb -csv_path results/imdb/crowd/results.csv
  ```

  
**Text Quality Metrics:**
- To obtain results for text quality metrics (e.g., fluency, cohesiveness, etc.), execute the following command:
  ```bash
  python eval_text_quality_metrics.py -eval_model EVAL_MODEL -task TASK -csv_path PATH_CSV -temperature TEMP
  ```
  - `TASK`: can be either "snli" or "imdb".
  - `PATH_CSV`:the path to the result .csv file.
  - `EVAL_MODEL`: Evaluation model, can be either "mistral" or "gpt". For GPT, you need to set up the `OPENAI_API_KEY`.
  - `TEMP`: the temperature for the evaluation model.

  - For example:
  ```bash
  python eval_text_quality_metrics.py -eval_model mistral -task snli -csv_path results/snli/crowd/results.csv -temperature 0.5
  ```

**Note:** The generated results required to reproduce the paper are available in the `results` folder. Simply utilize them and run the scripts to reproduce the paper.