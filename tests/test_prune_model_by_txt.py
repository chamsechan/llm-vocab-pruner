import unittest

from prune_model_by_txt import select_output_token_ids


class FakeGemmaTokenizer:
    all_special_ids = [0, 1, 2, 11, 12]

    def get_vocab(self):
        return {str(token_id): token_id for token_id in range(13)}


class OutputTokenSelectionTest(unittest.TestCase):
    def test_ignores_tokenizer_ids_outside_text_embedding(self):
        keep, ignored = select_output_token_ids(
            FakeGemmaTokenizer(), input_vocab_size=12, delete_ids={5}
        )
        self.assertEqual(ignored, [12])
        self.assertNotIn(5, keep)
        self.assertNotIn(12, keep)
        self.assertIn(11, keep)

    def test_rejects_deleting_valid_special_token(self):
        with self.assertRaisesRegex(ValueError, "special token"):
            select_output_token_ids(
                FakeGemmaTokenizer(), input_vocab_size=12, delete_ids={11}
            )


if __name__ == "__main__":
    unittest.main()
