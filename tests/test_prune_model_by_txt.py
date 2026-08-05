import unittest

import torch.nn as nn
from transformers import Qwen2Config

from prune_model_by_txt import select_output_token_ids, validate_exported_model


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


class FakeExportedModel:
    def __init__(self, tied=False, share_parameter=False):
        self.config = Qwen2Config(vocab_size=12, tie_word_embeddings=tied)
        self.config.output_vocab_size = 5
        self.input_embeddings = nn.Embedding(12, 4)
        self.output_embeddings = nn.Linear(4, 5, bias=False)
        if share_parameter:
            self.output_embeddings.weight = self.input_embeddings.weight

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings


class ExportValidationTest(unittest.TestCase):
    def test_accepts_untied_asymmetric_model(self):
        validate_exported_model(FakeExportedModel())

    def test_rejects_tie_flag(self):
        with self.assertRaisesRegex(RuntimeError, "tie_word_embeddings"):
            validate_exported_model(FakeExportedModel(tied=True))

    def test_rejects_shared_parameter(self):
        model = FakeExportedModel(share_parameter=True)
        model.config.output_vocab_size = 12
        with self.assertRaisesRegex(RuntimeError, "共享 Parameter"):
            validate_exported_model(model)


if __name__ == "__main__":
    unittest.main()
