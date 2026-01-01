import os
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model")

# Load tokenizer and model (FAST tokenizer is CRITICAL)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.eval()

def autocomplete(prompt, k=10):
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[1] + 3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=k
    )

    return list(
        set(tokenizer.decode(o, skip_special_tokens=True) for o in outputs)
    )
