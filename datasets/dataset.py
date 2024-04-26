import os

from datasets import load_dataset
def load_data(args):
    if args.dataset == "imdb":
        dataset = load_dataset("imdb")
        num_labels = 2
    return dataset, num_labels
