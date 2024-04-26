import pandas as pd
df = pd.read_csv("results/imdb/mice/white_box_imdb_2.csv",delimiter="\t").fillna(-1)
df = df.loc[df.groupby('data_idx')['minimality'].idxmin()]
df = df[["orig_input", "edited_input"]]
df = df.rename(columns={"orig_input": "orig_text",
                        "edited_input": "gen_text"})
df.to_csv("results/imdb/mice/results_2.csv", index=False)