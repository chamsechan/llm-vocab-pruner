import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from asymmetric_qwen2 import AsymmetricQwen2ForCausalLM
from asymmetric_qwen2 import convert_qwen2_to_asymmetric_vocab


class AsymmetricQwen2Test(unittest.TestCase):
    def make_model(self):
        config = Qwen2Config(
            vocab_size=12,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            tie_word_embeddings=True,
            bos_token_id=0,
            eos_token_id=11,
        )
        model = Qwen2ForCausalLM(config).eval()
        with torch.no_grad():
            values = torch.arange(12 * 16, dtype=torch.float32).reshape(12, 16)
            model.get_input_embeddings().weight.copy_(values / 100)
        return model

    def test_conversion_preserves_input_and_compacts_output(self):
        model = self.make_model()
        keep = [0, 2, 5, 7, 11]
        original_object = model.get_input_embeddings()
        original = original_object.weight.detach().clone()
        convert_qwen2_to_asymmetric_vocab(model, keep)

        self.assertIsInstance(model, AsymmetricQwen2ForCausalLM)
        self.assertIs(model.get_input_embeddings(), original_object)
        self.assertTrue(torch.equal(model.get_input_embeddings().weight, original))
        self.assertEqual(tuple(model.get_input_embeddings().weight.shape), (12, 16))
        self.assertEqual(tuple(model.get_output_embeddings().weight.shape), (5, 16))
        self.assertTrue(torch.equal(model.get_output_embeddings().weight, original[keep]))
        self.assertNotEqual(
            model.get_input_embeddings().weight.data_ptr(),
            model.get_output_embeddings().weight.data_ptr(),
        )

        logits = model(input_ids=torch.tensor([[1, 3, 10]])).logits
        self.assertEqual(tuple(logits.shape), (1, 3, 12))
        self.assertTrue(torch.isneginf(logits[..., [1, 3, 4, 6, 8, 9, 10]]).all())
        self.assertTrue(torch.isfinite(logits[..., keep]).all())

        # IDs 1/3/10 are valid multilingual inputs even though none can be output.
        generated = model.generate(
            input_ids=torch.tensor([[1, 3, 10]]),
            max_new_tokens=2,
            do_sample=False,
        )
        self.assertTrue(set(generated[0, 3:].tolist()).issubset(set(keep)))

    def test_save_and_auto_reload_round_trip(self):
        model = self.make_model()
        keep = [0, 2, 5, 7, 11]
        original = model.get_input_embeddings().weight.detach().clone()
        convert_qwen2_to_asymmetric_vocab(model, keep)

        with tempfile.TemporaryDirectory() as output_dir:
            model.save_pretrained(output_dir)
            loaded = AutoModelForCausalLM.from_pretrained(
                output_dir, trust_remote_code=True, local_files_only=True
            ).eval()
            self.assertEqual(type(loaded).__name__, "AsymmetricQwen2ForCausalLM")
            self.assertTrue(torch.equal(loaded.get_input_embeddings().weight, original))
            self.assertEqual(loaded.output_token_ids.tolist(), keep)
            self.assertEqual(tuple(loaded.get_output_embeddings().weight.shape), (5, 16))
            logits = loaded(input_ids=torch.tensor([[1, 3, 10]])).logits
            self.assertEqual(tuple(logits.shape), (1, 3, 12))
            self.assertTrue(torch.isneginf(logits[..., 1]).all())


if __name__ == "__main__":
    unittest.main()
