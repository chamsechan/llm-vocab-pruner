"""Causal-LM adapters with a full input vocabulary and compact output prefix."""

import copy
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from transformers.generation.logits_process import (
    LogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)


class _CompactRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """Apply repetition penalty only to IDs that exist in compact scores."""

    def __init__(self, penalty, prompt_ignore_length=None):
        self.penalty = penalty
        self.prompt_ignore_length = prompt_ignore_length

    def __call__(self, input_ids, scores):
        if self.prompt_ignore_length:
            input_ids = input_ids[:, self.prompt_ignore_length :]
        if scores.dim() != 2:
            raise RuntimeError(
                "Compact repetition penalty currently expects 2-D generation scores."
            )
        processed = scores.clone()
        for batch_index in range(scores.shape[0]):
            token_ids = input_ids[batch_index]
            token_ids = token_ids[
                (token_ids >= 0) & (token_ids < scores.shape[-1])
            ].unique()
            if token_ids.numel() == 0:
                continue
            token_scores = scores[batch_index, token_ids]
            penalized = torch.where(
                token_scores < 0,
                token_scores * self.penalty,
                token_scores / self.penalty,
            )
            processed[batch_index, token_ids] = penalized
        return processed


class _CompactSequenceBiasLogitsProcessor(SequenceBiasLogitsProcessor):
    """Allow input-only IDs in a bias prefix and drop unreachable final IDs."""

    def _prepare_bias_variables(self, scores):
        vocabulary_size = scores.shape[-1]
        self.sequence_bias = {
            sequence_ids: bias
            for sequence_ids, bias in self.sequence_bias.items()
            if sequence_ids[-1] < vocabulary_size
        }
        self.length_1_bias = torch.zeros(
            (vocabulary_size,), dtype=torch.float, device=scores.device
        )
        single_token_ids = []
        single_token_biases = []
        for sequence_ids, bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:
                single_token_ids.append(sequence_ids[0])
                single_token_biases.append(bias)
        if single_token_ids:
            self.length_1_bias[single_token_ids] = torch.tensor(
                single_token_biases, device=scores.device
            )
        self.prepared_bias_variables = True


class _CompactNoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Ignore banned next-token IDs outside the compact output prefix."""

    def __init__(self, ngram_size):
        self.ngram_size = ngram_size

    def __call__(self, input_ids, scores):
        current_length = input_ids.shape[-1]
        if current_length < self.ngram_size:
            return scores

        processed = scores.clone()
        prefix_size = self.ngram_size - 1
        for batch_index in range(input_ids.shape[0]):
            tokens = input_ids[batch_index].tolist()
            prefix = tuple(tokens[-prefix_size:]) if prefix_size else ()
            banned_ids = set()
            for start in range(current_length - self.ngram_size + 1):
                ngram = tokens[start : start + self.ngram_size]
                if tuple(ngram[:-1]) == prefix:
                    banned_ids.add(ngram[-1])
            valid_ids = [
                token_id
                for token_id in banned_ids
                if 0 <= token_id < scores.shape[-1]
            ]
            if valid_ids:
                processed[batch_index, valid_ids] = -float("inf")
        return processed


class _AsymmetricOutputVocabMixin:
    """Run a compact LM head whose IDs are the tokenizer's output prefix."""

    def _init_asymmetric_output(self, config):
        output_vocab_size = getattr(config, "output_vocab_size", None)
        if not isinstance(output_vocab_size, int) or output_vocab_size <= 0:
            raise ValueError("output_vocab_size must be a positive integer.")
        embedding = self.get_input_embeddings()
        input_vocab_size, hidden_size = embedding.weight.shape
        if output_vocab_size > input_vocab_size:
            raise ValueError("output_vocab_size cannot exceed the input vocabulary.")

        self.set_output_embeddings(
            nn.Linear(
                hidden_size,
                output_vocab_size,
                bias=False,
                device=embedding.weight.device,
                dtype=embedding.weight.dtype,
            )
        )
        config.tie_word_embeddings = False
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            text_config.tie_word_embeddings = False
        config.vocab_reordered = True

    def _get_logits_processor(self, *args, **kwargs):
        processors = super()._get_logits_processor(*args, **kwargs)
        for index, processor in enumerate(processors):
            if isinstance(processor, RepetitionPenaltyLogitsProcessor):
                processors[index] = _CompactRepetitionPenaltyLogitsProcessor(
                    processor.penalty, processor.prompt_ignore_length
                )
            elif isinstance(processor, NoRepeatNGramLogitsProcessor):
                processors[index] = _CompactNoRepeatNGramLogitsProcessor(
                    processor.ngram_size
                )
            elif isinstance(processor, SequenceBiasLogitsProcessor):
                processors[index] = _CompactSequenceBiasLogitsProcessor(
                    processor.sequence_bias
                )
        return processors

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            **kwargs,
        )
        if labels is not None:
            valid_labels = labels[labels != -100]
            if valid_labels.numel() and (
                valid_labels.min() < 0
                or valid_labels.max() >= self.config.output_vocab_size
            ):
                raise ValueError(
                    "labels contain input-only token IDs outside the compact "
                    "output vocabulary. Mask them with -100 or retain them in "
                    "the output prefix."
                )
            outputs.loss = self.loss_function(
                outputs.logits,
                labels,
                self.config.output_vocab_size,
            )
        return outputs


class AsymmetricQwen2ForCausalLM(_AsymmetricOutputVocabMixin, Qwen2ForCausalLM):
    """Qwen2/Qwen2.5 with a reordered full input and compact output prefix."""

    def __init__(self, config):
        config.tie_word_embeddings = False
        super().__init__(config)
        self._init_asymmetric_output(config)


class AsymmetricQwen3ForCausalLM(_AsymmetricOutputVocabMixin, Qwen3ForCausalLM):
    """Qwen3 with a reordered full input and compact output prefix."""

    def __init__(self, config):
        config.tie_word_embeddings = False
        super().__init__(config)
        self._init_asymmetric_output(config)


class AsymmetricQwen3_5ForConditionalGeneration(
    _AsymmetricOutputVocabMixin, Qwen3_5ForConditionalGeneration
):
    """Qwen3.5 with a reordered full text embedding and compact LM head."""

    def __init__(self, config):
        config.tie_word_embeddings = False
        config.text_config.tie_word_embeddings = False
        super().__init__(config)
        self._init_asymmetric_output(config)


class AsymmetricGemma3ForCausalLM(_AsymmetricOutputVocabMixin, Gemma3ForCausalLM):
    """Text-only Gemma 3 with a reordered full input and compact output prefix."""

    def __init__(self, config):
        config.tie_word_embeddings = False
        super().__init__(config)
        self._init_asymmetric_output(config)


_SUPPORTED_MODEL_ADAPTERS = (
    (Qwen2ForCausalLM, AsymmetricQwen2ForCausalLM, "AutoModelForCausalLM"),
    (Qwen3ForCausalLM, AsymmetricQwen3ForCausalLM, "AutoModelForCausalLM"),
    (
        Qwen3_5ForConditionalGeneration,
        AsymmetricQwen3_5ForConditionalGeneration,
        "AutoModelForImageTextToText",
    ),
    (Gemma3ForCausalLM, AsymmetricGemma3ForCausalLM, "AutoModelForCausalLM"),
)

_GENERATION_TOKEN_FIELDS = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "decoder_start_token_id",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "suppress_tokens",
    "begin_suppress_tokens",
    "bad_words_ids",
    "force_words_ids",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)


def _remap_token_value(value, old_to_new):
    if isinstance(value, int) and not isinstance(value, bool):
        return old_to_new.get(value, value)
    if isinstance(value, list):
        return [_remap_token_value(item, old_to_new) for item in value]
    if isinstance(value, tuple):
        return tuple(_remap_token_value(item, old_to_new) for item in value)
    return value


def _remap_config_token_ids(config, old_to_new):
    if config is None:
        return
    for field in _GENERATION_TOKEN_FIELDS:
        if hasattr(config, field):
            value = getattr(config, field)
            if value is not None:
                setattr(config, field, _remap_token_value(value, old_to_new))

    sequence_bias = getattr(config, "sequence_bias", None)
    if sequence_bias:
        config.sequence_bias = {
            _remap_token_value(key, old_to_new): bias
            for key, bias in sequence_bias.items()
        }

    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config is not config:
        _remap_config_token_ids(text_config, old_to_new)


def _remap_post_processor_ids(node, old_to_new):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "ids" and isinstance(value, list):
                node[key] = [_remap_token_value(item, old_to_new) for item in value]
            else:
                _remap_post_processor_ids(value, old_to_new)
    elif isinstance(node, list):
        for value in node:
            _remap_post_processor_ids(value, old_to_new)


def reorder_tokenizer(tokenizer, new_to_old: Iterable[int]):
    """Return a fast tokenizer whose in-range IDs follow ``new_to_old``.

    ``new_to_old[new_id]`` identifies the row/token at ``old_id``. Token text,
    BPE merges, normalizers, pre-tokenizers and decoding rules stay unchanged.
    Tokenizer-only IDs outside the input embedding range are left untouched.
    """

    if not getattr(tokenizer, "is_fast", False):
        raise TypeError("Vocabulary reordering requires a fast tokenizer.")

    new_to_old = list(new_to_old)
    if sorted(new_to_old) != list(range(len(new_to_old))):
        raise ValueError("new_to_old must be a permutation of embedding row IDs.")
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(new_to_old)}

    backend_data = json.loads(tokenizer.backend_tokenizer.to_str())
    model_vocab = backend_data.get("model", {}).get("vocab")
    if isinstance(model_vocab, dict):
        # Tokenizers such as Qwen keep special tokens only in added_tokens.
        # Once these IDs move into the output prefix, promote their token text
        # into the backend vocabulary as well; the added-token overlay still
        # preserves their special matching behavior.
        for added_token in backend_data.get("added_tokens", []):
            old_id = added_token.get("id")
            content = added_token.get("content")
            if old_id in old_to_new and content not in model_vocab:
                model_vocab[content] = old_id
        for token, old_id in model_vocab.items():
            model_vocab[token] = old_to_new.get(old_id, old_id)
    elif isinstance(model_vocab, list):
        if len(model_vocab) < len(new_to_old):
            raise ValueError(
                "Tokenizer model vocabulary is smaller than the embedding."
            )
        original_vocab = list(model_vocab)
        model_vocab[: len(new_to_old)] = [
            original_vocab[old_id] for old_id in new_to_old
        ]
    else:
        raise TypeError("Unsupported fast-tokenizer vocabulary representation.")

    for added_token in backend_data.get("added_tokens", []):
        old_id = added_token.get("id")
        if isinstance(old_id, int):
            added_token["id"] = old_to_new.get(old_id, old_id)

    _remap_post_processor_ids(backend_data.get("post_processor"), old_to_new)

    reordered = copy.deepcopy(tokenizer)
    reordered._tokenizer = Tokenizer.from_str(
        json.dumps(backend_data, ensure_ascii=False)
    )
    # Preserve the source tokenizer regex. Transformers 4.57.x callers can
    # also pass fix_mistral_regex=False to avoid its local-model false positive.
    reordered.init_kwargs["fix_mistral_regex"] = False
    vocab_file = getattr(reordered, "vocab_file", None)
    if isinstance(vocab_file, str) and vocab_file.endswith(".model"):
        # A SentencePiece file cannot express this backend BPE ID permutation
        # without a separate conversion. Do not export a stale slow-tokenizer
        # vocabulary beside the correct reordered tokenizer.json.
        reordered.vocab_file = None
        reordered.init_kwargs.pop("vocab_file", None)

    old_vocab = tokenizer.get_vocab()
    new_vocab = reordered.get_vocab()
    for token, old_id in old_vocab.items():
        expected_id = old_to_new.get(old_id, old_id)
        if new_vocab.get(token) != expected_id:
            raise RuntimeError(
                f"Tokenizer reordering failed for token {token!r}: "
                f"{new_vocab.get(token)} != {expected_id}."
            )
    return reordered


def _iter_token_ids(value):
    if isinstance(value, int) and not isinstance(value, bool):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_token_ids(item)


def _required_generation_token_ids(model):
    required_fields = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "decoder_start_token_id",
        "forced_bos_token_id",
        "forced_eos_token_id",
    )
    required = set()
    configs = [
        model.config,
        getattr(model.config, "text_config", None),
        getattr(model, "generation_config", None),
    ]
    for config in configs:
        if config is None:
            continue
        for field in required_fields:
            required.update(_iter_token_ids(getattr(config, field, None)))
    return required


def convert_to_reordered_output_vocab(model, tokenizer, keep_old_ids: Iterable[int]):
    """Reorder tokenizer/embedding and compact the untied LM head.

    Retained output tokens become the contiguous prefix ``[0, V_output)``.
    All other embedding rows remain available for input after that prefix. The
    returned model emits compact logits directly in the reordered tokenizer ID
    space, so no post-LM-head scatter or runtime ID mapping is required.
    """

    if getattr(model.config, "vocab_reordered", False):
        raise ValueError("The model vocabulary has already been reordered.")

    adapter = next(
        (
            (target_class, auto_class)
            for source_class, target_class, auto_class in _SUPPORTED_MODEL_ADAPTERS
            if isinstance(model, source_class)
        ),
        None,
    )
    if adapter is None:
        supported = ", ".join(
            source_class.__name__
            for source_class, _, _ in _SUPPORTED_MODEL_ADAPTERS
        )
        raise TypeError(
            f"Unsupported model class {type(model).__name__}. Supported: {supported}."
        )
    asymmetric_class, auto_class = adapter

    keep_old_ids = sorted(set(keep_old_ids))
    if not keep_old_ids:
        raise ValueError("At least one output token must be retained.")

    input_embedding = model.get_input_embeddings()
    input_vocab_size, hidden_dim = input_embedding.weight.shape
    if min(keep_old_ids) < 0 or max(keep_old_ids) >= input_vocab_size:
        raise ValueError("keep_old_ids contains an ID outside the input embedding.")

    keep_set = set(keep_old_ids)
    required_ids = {
        token_id
        for token_id in _required_generation_token_ids(model)
        if 0 <= token_id < input_vocab_size
    }
    missing_required = sorted(required_ids - keep_set)
    if missing_required:
        raise ValueError(
            "keep_old_ids must retain every configured BOS/EOS/PAD/forced "
            f"generation token; missing: {missing_required}."
        )

    keep_set = set(keep_old_ids)
    new_to_old = keep_old_ids + [
        old_id for old_id in range(input_vocab_size) if old_id not in keep_set
    ]
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(new_to_old)}

    source_output = model.get_output_embeddings()
    source_weight = (
        source_output.weight if source_output is not None else input_embedding.weight
    )
    keep_tensor = torch.tensor(
        keep_old_ids, dtype=torch.long, device=source_weight.device
    )
    compact_weight = source_weight.index_select(0, keep_tensor).clone()

    permutation = torch.tensor(
        new_to_old, dtype=torch.long, device=input_embedding.weight.device
    )
    reordered_embedding_weight = input_embedding.weight.index_select(
        0, permutation
    ).clone()
    input_embedding.weight = nn.Parameter(
        reordered_embedding_weight,
        requires_grad=input_embedding.weight.requires_grad,
    )

    compact_lm_head = nn.Linear(
        hidden_dim,
        len(keep_old_ids),
        bias=False,
        device=source_weight.device,
        dtype=source_weight.dtype,
    )
    compact_lm_head.weight = nn.Parameter(
        compact_weight, requires_grad=source_weight.requires_grad
    )
    model.set_output_embeddings(compact_lm_head)

    reordered_tokenizer = reorder_tokenizer(tokenizer, new_to_old)
    _remap_config_token_ids(model.config, old_to_new)
    _remap_config_token_ids(getattr(model, "generation_config", None), old_to_new)

    if input_embedding.padding_idx is not None:
        input_embedding.padding_idx = old_to_new.get(
            input_embedding.padding_idx, input_embedding.padding_idx
        )
    base_model = getattr(model, "model", None)
    if base_model is not None and hasattr(base_model, "padding_idx"):
        base_model.padding_idx = getattr(model.config, "pad_token_id", None)

    model.__class__ = asymmetric_class
    model.config.vocab_size = input_vocab_size
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        text_config.vocab_size = input_vocab_size
        text_config.tie_word_embeddings = False
    model.config.output_vocab_size = len(keep_old_ids)
    model.config.vocab_reordered = True
    model.config.tie_word_embeddings = False
    model.config.architectures = [asymmetric_class.__name__]

    asymmetric_class.register_for_auto_class(auto_class)
    return model, reordered_tokenizer
