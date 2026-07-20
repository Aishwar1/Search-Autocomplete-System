# QueryMind Research Lab

A real-time search autocomplete system with 9 ML visualization modules.

## Stack
- **Backend:** Python / Flask
- **ML:** PyTorch, Hugging Face Transformers (GPT-2), scikit-learn
- **Frontend:** Vanilla JS, D3.js, Three.js, Chart.js

## How to Run

```bash
python app.py
```

App serves on port 5000. The transformer model is lazy-loaded on first request.

### Optional: Train the Fine-Tuned Model
```bash
python model/train.py
```
This produces `model/final_model/`. Without it, the app falls back to base GPT-2.

## Tabs
| Tab | What it shows |
|---|---|
| Search Engine | Live autocomplete combining Trie + Markov + GPT-2 |
| Trie Explorer | Prefix tree visualization, O(k) lookup |
| Markov Chain | Bigram/trigram transition graph + probability bars |
| LSTM Internals | Character-level LSTM gate activations (real numpy math) |
| 3D Embeddings | GPT-2 token vectors projected to 3D via PCA |
| Gradient Descent | Rosenbrock loss surface animation |
| Decision Trees | GBDT ranker — feature importance + boosting curve |
| User History | Session query log + analytics |
| Metrics | Training loss/perplexity curves |

## Deployment (Render)

The project is Render-ready:
- `Procfile`: `web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- `render.yaml`: complete service config (auto-generates SESSION_SECRET)
- `SESSION_SECRET` env var is read from environment (falls back to a dev key)

**Important:** Use 1 worker on Render — PyTorch models are large and multi-worker would OOM on free/starter plans.

## Project Structure
```
app.py              # Flask app, all API routes
model/
  inference.py      # GPT-2 autocomplete + attention
  embeddings.py     # Word vectors, PCA, LSTM simulation, gradient surface
  gbdt.py           # GBDT ranker + feature extraction
  markov.py         # Markov chain (bigram/trigram)
  trie.py           # Prefix trie
  session_store.py  # In-memory session history
  train.py          # Fine-tuning pipeline (optional)
  data/queries.txt  # Training corpus
static/
  viz_*.js          # One JS file per visualization tab
  style.css         # Dark theme
  main.js           # Tab controller + search engine logic
templates/index.html
```

## User Preferences
- Keep the existing project structure and stack — do not restructure or migrate.
