import unittest


from metrics import Metrics

class TestScoreMinimality(unittest.TestCase):
    def setUp(self):
        self.metrics = Metrics()  # Replace with the actual class instance

    def test_normalized_score(self):
        orig_sent = "The quick brown fox jumps over the lazy dog."
        edited_sent = "A quick brown fox leaps over a lazy dog."
        expected_normalized_score = 0.3  # Calculate the expected normalized score here

        calculated_score = self.metrics.score_minimality(orig_sent, edited_sent, normalized=True)
        self.assertAlmostEqual(expected_normalized_score, calculated_score, places=2)

    def test_raw_score(self):
        orig_sent = "Hello, world!"
        edited_sent = "Hello, brave new world!"
        expected_raw_score = 2  # Calculate the expected raw score here

        calculated_score = self.metrics.score_minimality(orig_sent, edited_sent, normalized=False)
        self.assertEqual(expected_raw_score, calculated_score)

class TestFlipRate(unittest.TestCase):
    def setUp(self):
        self.metrics = Metrics()
    def test_same_labels(self):
        original_labels = [1, 2, 3, 4, 5]
        new_labels = [1, 2, 3, 4, 5]
        self.assertEqual(self.metrics.flip_rate(original_labels, new_labels), 0.0)

    def test_one_different_label(self):
        original_labels = [1, 2, 3, 4, 5]
        new_labels = [1, 2, 3, 6, 5]
        self.assertEqual(self.metrics.flip_rate(original_labels, new_labels), 0.2)

    def test_empty_lists(self):
        self.assertEqual(self.metrics.flip_rate([], []), 0.0)

    def test_different_lengths(self):
        original_labels = [1, 2, 3, 4, 5]
        with self.assertRaises(ValueError):
            self.metrics.flip_rate(original_labels, [1, 2, 3, 4])

class TestScorePerplexity(unittest.TestCase):
    def setUp(self):
        self.metrics = Metrics()
    def test_perplexity_score(self):
        # Test case 1: Empty input
        # sents_empty = []
        # self.assertRaises(ValueError, self.metrics.score_perplexity, sents_empty)

        # Test case 2: Single sentence
        sents_single = ["This is a test sentence."]
        perplexity_single = self.metrics.score_perplexity(sents_single)
        # self.assertIsInstance(perplexity_single, float)
        self.assertGreaterEqual(perplexity_single, 0.0)

        # Test case 3: Multiple sentences
        sents_multiple = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
        perplexity_multiple = self.metrics.score_perplexity(sents_multiple)
        # self.assertIsInstance(perplexity_multiple, float)
        self.assertGreaterEqual(perplexity_multiple, 0.0)

if __name__ == '__main__':
    unittest.main()