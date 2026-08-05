"""Qwen2 with a full input vocabulary and a compact, output-only LM head."""

from typing import Iterable, Optional

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


class AsymmetricQwen2ForCausalLM(Qwen2ForCausalLM):
    """Keep original token IDs for input while restricting generated tokens.

    The LM head computes only ``config.output_token_ids``. Its compact logits are
    scattered back to the original vocabulary positions so standard generation
    and the original multilingual tokenizer continue to use original token IDs.
    """

    def __init__(self, config):
        output_token_ids = getattr(
            config, "output_token_ids", list(range(config.vocab_size))
        )
        if not output_token_ids:
            raise ValueError("output_token_ids must contain at least one token ID.")
        if min(output_token_ids) < 0 or max(output_token_ids) >= config.vocab_size:
            raise ValueError("output_token_ids contains an ID outside vocab_size.")

        config.tie_word_embeddings = False
        super().__init__(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            len(output_token_ids),
            bias=False,
            device=self.get_input_embeddings().weight.device,
            dtype=self.get_input_embeddings().weight.dtype,
        )
        self.register_buffer(
            "output_token_ids",
            torch.as_tensor(output_token_ids, dtype=torch.long),
            persistent=True,
        )
        config.output_vocab_size = len(output_token_ids)
        config.output_token_ids = list(output_token_ids)
        config.tie_word_embeddings = False

    def _expand_output_logits(self, compact_logits):
        full_shape = (*compact_logits.shape[:-1], self.config.vocab_size)
        full_logits = compact_logits.new_full(full_shape, -torch.inf)
        token_ids = self.output_token_ids
        if token_ids.device != compact_logits.device:
            token_ids = token_ids.to(compact_logits.device)
        full_logits.index_copy_(-1, token_ids, compact_logits)
        return full_logits

    def forward(self, *args, labels: Optional[torch.LongTensor] = None, **kwargs):
        # Compute loss after expanding logits because the compact head indices are
        # deliberately not tokenizer IDs.
        outputs = super().forward(*args, labels=None, **kwargs)
        outputs.logits = self._expand_output_logits(outputs.logits)
        if labels is not None:
            outputs.loss = self.loss_function(
                logits=outputs.logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        return outputs


def convert_qwen2_to_asymmetric_vocab(model, keep_old_ids: Iterable[int]):
    """Keep input embedding/tokenizer intact and independently prune the LM head."""

    if not isinstance(model, Qwen2ForCausalLM):
        raise TypeError(
            "Asymmetric input/output vocabulary export currently supports Qwen2 "
            f"models, got {type(model).__name__}."
        )

    keep_old_ids = sorted(set(keep_old_ids))
    if not keep_old_ids:
        raise ValueError("At least one output token must be retained.")

    input_embedding = model.get_input_embeddings()
    input_vocab_size, hidden_dim = input_embedding.weight.shape
    if min(keep_old_ids) < 0 or max(keep_old_ids) >= input_vocab_size:
        raise ValueError("keep_old_ids contains an ID outside the input embedding.")

    source_output = model.get_output_embeddings()
    source_weight = (
        source_output.weight if source_output is not None else input_embedding.weight
    )
    keep_tensor = torch.tensor(
        keep_old_ids, dtype=torch.long, device=source_weight.device
    )
    compact_weight = source_weight.index_select(0, keep_tensor).clone()
    compact_lm_head = nn.Linear(
        hidden_dim,
        len(keep_old_ids),
        bias=False,
        device=source_weight.device,
        dtype=source_weight.dtype,
    )
    compact_lm_head.weight = nn.Parameter(compact_weight)

    # The embedding object and its weight are intentionally never replaced.
    model.set_output_embeddings(compact_lm_head)
    model.__class__ = AsymmetricQwen2ForCausalLM
    model.register_buffer(
        "output_token_ids", keep_tensor.clone(), persistent=True
    )

    model.config.vocab_size = input_vocab_size
    model.config.output_vocab_size = len(keep_old_ids)
    model.config.output_token_ids = keep_old_ids
    model.config.tie_word_embeddings = False
    model.config.architectures = ["AsymmetricQwen2ForCausalLM"]

    AsymmetricQwen2ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    return model
