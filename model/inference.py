"""
Transformer Inference — GPT-2 Autocomplete
==========================================
Loads the fine-tuned (or base) GPT-2 model and generates
top-K query completions with confidence scores and attention weights.
"""

import os
import json
import numpy as np

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINE_TUNED_PATH = os.path.join(BASE_DIR, 'final_model')
MODEL_NAME = 'gpt2'   # fallback

# ── Load once ────────────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_is_finetuned = False


def _load():
    global _model, _tokenizer, _is_finetuned

    if os.path.isdir(FINE_TUNED_PATH):
        try:
            _tokenizer = GPT2TokenizerFast.from_pretrained(FINE_TUNED_PATH)
            _model = GPT2LMHeadModel.from_pretrained(FINE_TUNED_PATH)
            _is_finetuned = True
            print('[Inference] Loaded fine-tuned model.')
            return
        except Exception as e:
            print(f'[Inference] Fine-tuned load failed: {e}. Falling back to base GPT-2.')

    _tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    _model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    _is_finetuned = False
    print('[Inference] Loaded base GPT-2 model.')


def load_model():
    """Public entry point for app.py compatibility."""
    if _model is None:
        _load()
    return _model


def get_model_and_tokenizer():
    if _model is None:
        _load()
    return _model, _tokenizer


# ── Autocomplete ─────────────────────────────────────────────────────────────
def autocomplete(query: str, k: int = 8):
    model, tokenizer = get_model_and_tokenizer()
    model.eval()

    inputs = tokenizer(query, return_tensors='pt')
    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.85,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    suggestions = []
    raw_scores = []
    scores_tensor = torch.stack(outputs.scores, dim=1)  # (k, gen_len, vocab)

    for i, seq in enumerate(outputs.sequences):
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        completion = decoded[len(query):].strip()
        if not completion:
            continue

        token_probs = scores_tensor[i].softmax(dim=-1)
        score = token_probs.max(dim=-1).values.mean().item()

        raw_scores.append(score)
        suggestions.append({
            'text': query.strip() + ' ' + completion,
            'confidence': score,
            'model': 'transformer'
        })

    total = sum(raw_scores) or 1.0
    for s in suggestions:
        s['confidence'] = round(s['confidence'] / total, 4)

    return {
        'suggestions': suggestions[:k],
        'tokens': tokenizer.tokenize(query),
        'is_finetuned': _is_finetuned,
        'model': 'fine-tuned GPT-2' if _is_finetuned else 'base GPT-2'
    }


# ── Attention Extraction ─────────────────────────────────────────────────────
def get_attention_weights(query: str, layer: int = 5):
    """Extract real multi-head attention weights from GPT-2."""
    model, tokenizer = get_model_and_tokenizer()
    model.eval()

    inputs = tokenizer(query, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    tokens = [t.replace('Ġ', ' ').strip() for t in tokens]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of (num_layers,) tensors
    # each: (1, num_heads, seq_len, seq_len)
    num_layers = len(outputs.attentions)
    layer = max(0, min(layer, num_layers - 1))

    attn_all_layers = []
    for l_idx, layer_attn in enumerate(outputs.attentions):
        avg_head = layer_attn[0].mean(dim=0).tolist()           # (seq, seq)
        heads = [h.tolist() for h in layer_attn[0][:4]]         # first 4 heads
        attn_all_layers.append({
            'layer': l_idx,
            'avg': [[round(v, 4) for v in row] for row in avg_head],
            'heads': [[[round(v, 4) for v in row] for row in h] for h in heads]
        })

    return {
        'tokens': tokens,
        'attention_by_layer': attn_all_layers,
        'selected_layer': layer,
        'num_layers': num_layers,
        'num_heads': 12,
        'seq_len': len(tokens)
    }


# ── Metrics ──────────────────────────────────────────────────────────────────
def get_metrics():
    metrics_path = os.path.join(BASE_DIR, 'metrics.json')
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)

    # Return synthetic metrics when model hasn't been trained yet
    import math
    steps = 40
    loss_curve = [3.8 * math.exp(-0.06 * i) + 0.3 + 0.05 * (i % 3) for i in range(steps)]
    perplexity = [math.exp(l) for l in loss_curve]
    return {
        'epochs': 5,
        'batch_size': 2,
        'learning_rate': 5e-5,
        'loss_curve': [round(l, 4) for l in loss_curve],
        'perplexity': [round(p, 2) for p in perplexity],
        'demo_mode': True
    }
