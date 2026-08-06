import json
import os
import tempfile
import unittest

import torch
from accelerate import init_empty_weights
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3TextConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
    Gemma3ForCausalLM,
)

from asymmetric_models import (
    AsymmetricGemma3ForCausalLM,
    AsymmetricQwen2ForCausalLM,
    convert_to_reordered_output_vocab,
)


class AsymmetricModelContract:
    asymmetric_class = None

    def make_model(self):
        raise NotImplementedError

    def make_tokenizer(self):
        vocab = {f"t{token_id}": token_id for token_id in range(12)}
        backend = Tokenizer(models.WordLevel(vocab, unk_token="t1"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="t1",
            pad_token="t0",
            bos_token="t2",
            eos_token="t11",
        )

    def test_conversion_reorders_embedding_and_emits_compact_logits(self):
        model = self.make_model()
        tokenizer = self.make_tokenizer()
        keep = [0, 2, 5, 7, 11]
        permutation = keep + [1, 3, 4, 6, 8, 9, 10]
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(permutation)}
        original_object = model.get_input_embeddings()
        original = original_object.weight.detach().clone()
        old_input_ids = torch.tensor([[1, 3, 10]])
        with torch.no_grad():
            reference_logits = model(input_ids=old_input_ids).logits[..., keep]

        model, tokenizer = convert_to_reordered_output_vocab(model, tokenizer, keep)

        self.assertIsInstance(model, self.asymmetric_class)
        self.assertFalse(model.config.tie_word_embeddings)
        self.assertTrue(model.config.vocab_reordered)
        self.assertFalse(hasattr(model.config, "output_token_ids"))
        self.assertFalse(hasattr(model, "output_token_ids"))
        self.assertIs(model.get_input_embeddings(), original_object)
        self.assertTrue(
            torch.equal(model.get_input_embeddings().weight, original[permutation])
        )
        self.assertEqual(tuple(model.get_input_embeddings().weight.shape), (12, 16))
        self.assertEqual(tuple(model.get_output_embeddings().weight.shape), (5, 16))
        self.assertTrue(
            torch.equal(model.get_output_embeddings().weight, original[keep])
        )
        self.assertNotEqual(
            model.get_input_embeddings().weight.data_ptr(),
            model.get_output_embeddings().weight.data_ptr(),
        )

        for old_id in range(12):
            self.assertEqual(
                tokenizer.convert_tokens_to_ids(f"t{old_id}"), old_to_new[old_id]
            )
        self.assertEqual(tokenizer.eos_token_id, old_to_new[11])
        self.assertEqual(model.config.eos_token_id, old_to_new[11])

        new_input_ids = torch.tensor(
            [[old_to_new[token_id] for token_id in old_input_ids[0].tolist()]]
        )
        logits = model(input_ids=new_input_ids).logits
        self.assertEqual(tuple(logits.shape), (1, 3, 5))
        self.assertTrue(torch.allclose(logits, reference_logits, atol=1e-6, rtol=0))

        generated = model.generate(
            input_ids=new_input_ids,
            attention_mask=torch.ones_like(new_input_ids),
            pad_token_id=0,
            max_new_tokens=2,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            bad_words_ids=[[5, 1]],
        )
        self.assertTrue((generated[0, 3:] < len(keep)).all())

        with self.assertRaisesRegex(ValueError, "input-only token IDs"):
            model(input_ids=new_input_ids, labels=torch.tensor([[0, 6, 1]]))

    def test_conversion_rejects_missing_required_generation_token(self):
        with self.assertRaisesRegex(ValueError, "BOS/EOS/PAD/forced"):
            convert_to_reordered_output_vocab(
                self.make_model(), self.make_tokenizer(), [0, 2, 5, 7]
            )

    def test_save_and_auto_reload_round_trip(self):
        model = self.make_model()
        tokenizer = self.make_tokenizer()
        keep = [0, 2, 5, 7, 11]
        permutation = keep + [1, 3, 4, 6, 8, 9, 10]
        original = model.get_input_embeddings().weight.detach().clone()
        model, tokenizer = convert_to_reordered_output_vocab(model, tokenizer, keep)

        with tempfile.TemporaryDirectory() as output_dir:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            with open(
                os.path.join(output_dir, "config.json"), encoding="utf-8"
            ) as config_file:
                saved_config = json.load(config_file)
            self.assertFalse(saved_config["tie_word_embeddings"])
            self.assertTrue(saved_config["vocab_reordered"])
            self.assertNotIn("output_token_ids", saved_config)
            self.assertEqual(saved_config["output_vocab_size"], len(keep))
            self.assertIn(
                "asymmetric_models.Asymmetric",
                saved_config["auto_map"]["AutoModelForCausalLM"],
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "asymmetric_models.py"))
            )

            loaded_tokenizer = AutoTokenizer.from_pretrained(
                output_dir, local_files_only=True
            )
            loaded = AutoModelForCausalLM.from_pretrained(
                output_dir, trust_remote_code=True, local_files_only=True
            ).eval()
            self.assertEqual(type(loaded).__name__, self.asymmetric_class.__name__)
            self.assertFalse(loaded.config.tie_word_embeddings)
            self.assertTrue(loaded.config.vocab_reordered)
            self.assertTrue(
                torch.equal(loaded.get_input_embeddings().weight, original[permutation])
            )
            self.assertEqual(
                tuple(loaded.get_output_embeddings().weight.shape), (5, 16)
            )
            self.assertEqual(loaded_tokenizer.convert_tokens_to_ids("t11"), 4)
            self.assertNotEqual(
                loaded.get_input_embeddings().weight.data_ptr(),
                loaded.get_output_embeddings().weight.data_ptr(),
            )
            logits = loaded(input_ids=torch.tensor([[5, 6, 11]])).logits
            self.assertEqual(tuple(logits.shape), (1, 3, 5))


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
            bos_token_id=2,
            eos_token_id=11,
            pad_token_id=0,
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
            bos_token_id=2,
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
            tie_word_embeddings=False,
        )
        config.output_vocab_size = 5
        config.vocab_reordered = True
        with init_empty_weights():
            model = AsymmetricGemma3ForCausalLM(config)

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
