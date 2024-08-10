
## Method-Specific Environments

Each method has its own environment. Please ensure you check the setup for each method accordingly.

### LLAMA-2

To generate the counterfactual with LLAMA-2, use the following command:

```bash
python methods/LLAMA-2/simple_prompt.py -task TASK -batch_size SIZE -temperature TEMP
```

- **`TASK`**: The task to be performed, either `IMDB` or `SNLI`.
- **`SIZE`**: The batch size used during generation.
- **`TEMP`**: The temperature setting for LLAMA-2.
