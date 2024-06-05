from spacy.lang.en import English
import nltk
from evaluate import load
class Metrics:
    def __init__(self):
        self.perplexity = load("perplexity", module_type="metric")
        nlp = English()
        self.tokenizer = nlp.tokenizer

    def score_perplexity(self, sents: list[str], model = "gpt2"):
        """
        Calculate the perplexity score, measures how likely the model is to generate the input text sequence.
        More information: https://huggingface.co/docs/transformers/perplexity
        :param sents:
        :return:
        """
        
        results = self.perplexity.compute(predictions=sents, model_id=model, max_length=1024, device="cuda")
        return results
    

    def score_minimality(self, orig_sent: str, edited_sent: str, normalized: bool = False) -> float:
        """
          Calculate Levenshtein distance(token-level) indicating the minimality of changes between two sentences.

          This method takes in an original sentence and an edited sentence, both as strings.
          It calculates the Levenshtein edit distance between the tokenized versions of these sentences,
          representing the minimum number of single-token edits needed to transform one into the other.

          Parameters:
          - orig_sent (str): The original sentence before editing.
          - edited_sent (str): The edited version of the sentence.
          - normalized (bool, optional): If True, returns a normalized score relative to the length of
            the original sentence. If False, returns the raw edit distance value.

          Returns:
          - float: The calculated minimality score. If 'normalized' is True, the score represents the
            proportion of changes relative to the original sentence length.u

            Source:
          """
        
        
        tokenized_original = [t.text for t in self.tokenizer(orig_sent)]
        tokenized_edited = [t.text for t in self.tokenizer(edited_sent)]
        levenshtein_dist = nltk.edit_distance(tokenized_original, tokenized_edited)
        if normalized:
            return levenshtein_dist / len(tokenized_original)
        else:
            return levenshtein_dist
    @staticmethod
    def flip_rate(original_labels: list, new_labels: list) -> float:
        """
        Calculate the rate of label differences between original_labels and new_labels.

        This function takes two lists of labels, `original_labels` and `new_labels`, and calculates the rate at which the labels
        have changed between the two lists. It does so by counting the number of differing labels and dividing it by the total
        number of labels in the lists.

        :param original_labels: A list of original labels.
        :param new_labels: A list of new labels to compare against the original labels.
        :return: The rate of label differences as a floating-point number between 0 and 1.
        """
        if len(original_labels) != len(new_labels):
            raise ValueError("Input lists must have the same length.")

        num_differences = sum(1 for orig, new in zip(original_labels, new_labels) if orig != new)
        total_labels = len(original_labels)

        flip_rate = num_differences / total_labels
        return flip_rate