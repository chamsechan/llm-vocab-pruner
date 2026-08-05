import json
import os
import tempfile
import unittest

import torch
from accelerate import init_empty_weights
from transformers import (
    AutoModelForCausalLM,
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from asymmetric_models import (
    AsymmetricGemma3ForCausalLM,
    AsymmetricQwen2ForCausalLM,
    convert_to_asymmetric_output_vocab,
)


class AsymmetricModelContract:
    asymmetric_class = None

    def make_model(self):
        raise NotImplementedError

    def test_conversion_generation_and_output_mask(self):
        model = self.make_model()
        keep = [0, 2, 5, 7, 11]
        original_object = model.get_input_embeddings()
        original = original_object.weight.detach().clone()
        convert_to_asymmetric_output_vocab(model, keep)

        self.assertIsInstance(model, self.asymmetric_class)
        self.assertFalse(model.config.tie_word_embeddings)
        self.assertIs(model.get_input_embeddings(), original_object)
        self.assertTrue(torch.equal(model.get_input_embeddings().weight, original))
        self.assertEqual(tuple(model.get_input_embeddings().weight.shape), (12, 16))
        self.assertEqual(tuple(model.get_output_embeddings().weight.shape), (5, 16))
        self.assertTrue(torch.equal(model.get_output_embeddings().weight, original[keep]))
        self.assertNotEqual(
            model.get_input_embeddings().weight.data_ptr(),
            model.get_output_embeddings().weight.data_ptr(),
        )

        input_ids = torch.tensor([[1, 3, 10]])
        logits = model(input_ids=input_ids).logits
        self.assertEqual(tuple(logits.shape), (1, 3, 12))
        self.assertTrue(torch.isneginf(logits[..., [1, 3, 4, 6, 8, 9, 10]]).all())
        self.assertTrue(torch.isfinite(logits[..., keep]).all())

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=0,
            max_new_tokens=2,
            do_sample=False,
        )
        self.assertTrue(set(generated[0, 3:].tolist()).issubset(set(keep)))

    def test_save_and_auto_reload_round_trip(self):
        model = self.make_model()
        keep = [0, 2, 5, 7, 11]
        original = model.get_input_embeddings().weight.detach().clone()
        convert_to_asymmetric_output_vocab(model, keep)

        with tempfile.TemporaryDirectory() as output_dir:
            model.save_pretrained(output_dir)
            with open(
                os.path.join(output_dir, "config.json"), encoding="utf-8"
            ) as config_file:
                saved_config = json.load(config_file)
            self.assertFalse(saved_config["tie_word_embeddings"])
            self.assertEqual(saved_config["output_token_ids"], keep)
            self.assertEqual(saved_config["output_vocab_size"], len(keep))
            self.assertIn(
                "asymmetric_models.Asymmetric",
                saved_config["auto_map"]["AutoModelForCausalLM"],
            )
            self.assertTrue(os.path.exists(os.path.join(output_dir, "asymmetric_models.py")))

            loaded = AutoModelForCausalLM.from_pretrained(
                output_dir, trust_remote_code=True, local_files_only=True
            ).eval()
            self.assertEqual(type(loaded).__name__, self.asymmetric_class.__name__)
            self.assertFalse(loaded.config.tie_word_embeddings)
            self.assertTrue(torch.equal(loaded.get_input_embeddings().weight, original))
            self.assertEqual(loaded.output_token_ids.tolist(), keep)
            self.assertEqual(tuple(loaded.get_output_embeddings().weight.shape), (5, 16))
            self.assertNotEqual(
                loaded.get_input_embeddings().weight.data_ptr(),
                loaded.get_output_embeddings().weight.data_ptr(),
            )
            logits = loaded(input_ids=torch.tensor([[1, 3, 10]])).logits
            self.assertEqual(tuple(logits.shape), (1, 3, 12))
            self.assertTrue(torch.isneginf(logits[..., 1]).all())


class AsymmetricQwen2Test(AsymmetricModelContract, unittest.TestCase):
    asymmetric_class = AsymmetricQwen2ForCausalLM

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
        return make_deterministic_model(Qwen2ForCausalLM(config))


class AsymmetricGemma3Test(AsymmetricModelContract, unittest.TestCase):
    asymmetric_class = AsymmetricGemma3ForCausalLM

    def make_model(self):
        config = Gemma3TextConfig(
            vocab_size=12,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            sliding_window=16,
            layer_types=["full_attention"],
            tie_word_embeddings=True,
            bos_token_id=0,
            eos_token_id=11,
            pad_token_id=0,
        )
        return make_deterministic_model(Gemma3ForCausalLM(config))

    def test_gemma3_270m_production_shapes_on_meta_device(self):
        layer_types = [
            "full_attention" if (index + 1) % 6 == 0 else "sliding_attention"
            for index in range(18)
        ]
        config = Gemma3TextConfig(
            vocab_size=262144,
            hidden_size=640,
            intermediate_size=2048,
            num_hidden_layers=18,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=256,
            max_position_embeddings=32768,
            sliding_window=512,
            layer_types=layer_types,
            query_pre_attn_scalar=256,
            tie_word_embeddings=True,
        )
        with init_empty_weights():
            model = Gemma3ForCausalLM(config)
            original_embedding = model.get_input_embeddings()
            convert_to_asymmetric_output_vocab(model, [0, 1, 2, 100, 200])

        self.assertIsInstance(model, AsymmetricGemma3ForCausalLM)
        self.assertIs(model.get_input_embeddings(), original_embedding)
        self.assertEqual(
            tuple(model.get_input_embeddings().weight.shape), (262144, 640)
        )
        self.assertEqual(tuple(model.get_output_embeddings().weight.shape), (5, 640))


def make_deterministic_model(model):
    model.eval()
    with torch.no_grad():
        values = torch.arange(12 * 16, dtype=torch.float32).reshape(12, 16)
        model.get_input_embeddings().weight.copy_(values / 100)
    return model


if __name__ == "__main__":
    unittest.main()
