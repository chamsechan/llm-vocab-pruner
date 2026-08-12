import json
import os
import tempfile
import unittest

import torch
from accelerate import init_empty_weights
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    Gemma3TextConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
    Gemma3ForCausalLM,
)

from asymmetric_models import (
    AsymmetricGemma3ForCausalLM,
    AsymmetricQwen2ForCausalLM,
    AsymmetricQwen3ForCausalLM,
    AsymmetricQwen3_5ForConditionalGeneration,
    convert_to_reordered_output_vocab,
)


class AsymmetricModelContract:
    asymmetric_class = None
    auto_model_class = AutoModelForCausalLM

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
        token_config = getattr(model.config, "text_config", model.config)
        self.assertEqual(token_config.eos_token_id, old_to_new[11])

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
            auto_class_name = self.auto_model_class.__name__
            self.assertIn(
                "asymmetric_models.Asymmetric",
                saved_config["auto_map"][auto_class_name],
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "asymmetric_models.py"))
            )

            loaded_tokenizer = AutoTokenizer.from_pretrained(
                output_dir, local_files_only=True
            )
            loaded = self.auto_model_class.from_pretrained(
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


class AsymmetricQwen3Test(AsymmetricModelContract, unittest.TestCase):
    asymmetric_class = AsymmetricQwen3ForCausalLM

    def make_model(self):
        config = Qwen3Config(
            vocab_size=12,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            tie_word_embeddings=True,
            bos_token_id=2,
            eos_token_id=11,
            pad_token_id=0,
        )
        return make_deterministic_model(Qwen3ForCausalLM(config))

    def test_qwen3_0_6b_production_shapes_on_meta_device(self):
        config = Qwen3Config(
            vocab_size=151936,
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=40960,
            tie_word_embeddings=False,
        )
        config.output_vocab_size = 5
        config.vocab_reordered = True
        with init_empty_weights():
            model = AsymmetricQwen3ForCausalLM(config)

        self.assertEqual(
            tuple(model.get_input_embeddings().weight.shape), (151936, 1024)
        )
        self.assertEqual(tuple(model.get_output_embeddings().weight.shape), (5, 1024))


class AsymmetricQwen3_5Test(AsymmetricModelContract, unittest.TestCase):
    asymmetric_class = AsymmetricQwen3_5ForConditionalGeneration
    auto_model_class = AutoModelForImageTextToText

    def make_model(self):
        text_config = Qwen3_5TextConfig(
            vocab_size=12,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            layer_types=["full_attention"],
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.75,
                "mrope_section": [1, 1, 1],
                "mrope_interleaved": True,
            },
            tie_word_embeddings=True,
            bos_token_id=2,
            eos_token_id=11,
            pad_token_id=0,
        )
        vision_config = Qwen3_5VisionConfig(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=16,
            num_position_embeddings=16,
            patch_size=2,
            spatial_merge_size=1,
            temporal_patch_size=1,
        )
        config = Qwen3_5Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=6,
            video_token_id=8,
            vision_start_token_id=9,
            vision_end_token_id=10,
            tie_word_embeddings=True,
        )
        return make_deterministic_model(
            Qwen3_5ForConditionalGeneration(config)
        )

    def test_qwen3_5_0_8b_production_shapes_on_meta_device(self):
        layer_types = [
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(24)
        ]
        text_config = Qwen3_5TextConfig(
            vocab_size=248320,
            hidden_size=1024,
            intermediate_size=3584,
            num_hidden_layers=24,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            max_position_embeddings=262144,
            layer_types=layer_types,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
            tie_word_embeddings=True,
            eos_token_id=248044,
        )
        vision_config = Qwen3_5VisionConfig(
            depth=12,
            hidden_size=768,
            intermediate_size=3072,
            num_heads=12,
            out_hidden_size=1024,
            num_position_embeddings=2304,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )
        config = Qwen3_5Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=248056,
            video_token_id=248057,
            vision_start_token_id=248053,
            vision_end_token_id=248054,
            tie_word_embeddings=True,
        )
        config.output_vocab_size = 5
        config.vocab_reordered = True
        with init_empty_weights():
            model = AsymmetricQwen3_5ForConditionalGeneration(config)

        self.assertEqual(
            tuple(model.get_input_embeddings().weight.shape), (248320, 1024)
        )
        self.assertEqual(tuple(model.get_output_embeddings().weight.shape), (5, 1024))
        self.assertFalse(model.config.tie_word_embeddings)
        self.assertFalse(model.config.text_config.tie_word_embeddings)


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
        rows, columns = model.get_input_embeddings().weight.shape
        values = torch.arange(
            rows * columns, dtype=torch.float32
        ).reshape(rows, columns)
        model.get_input_embeddings().weight.copy_(values / 100)
    return model


if __name__ == "__main__":
    unittest.main()
