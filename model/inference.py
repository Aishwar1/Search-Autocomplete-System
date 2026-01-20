import os
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# -----------------------------
# Load model + tokenizer ONCE
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model")

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.eval()


# -----------------------------
# Autocomplete (MULTI-WORD)
# -----------------------------
def autocomplete(query, k=10):
    inputs = tokenizer(query, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    suggestions = []
    raw_scores = []

    scores = torch.stack(outputs.scores, dim=1)

    for i, seq in enumerate(outputs.sequences):
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        completion = decoded[len(query):].strip()

        if completion:
            # Use mean log-prob as confidence proxy
            token_scores = scores[i].softmax(dim=-1)
            score = token_scores.max().item()

            raw_scores.append(score)
            suggestions.append({
                "text": query + " " + completion,
                "confidence": score
            })

    # 🔥 NORMALIZE to sum to 1
    total = sum(raw_scores) or 1.0
    for s in suggestions:
        s["confidence"] = s["confidence"] / total

    return {
        "suggestions": suggestions[:k],
        "tokens": tokenizer.tokenize(query)
    }
