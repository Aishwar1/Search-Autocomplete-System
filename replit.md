# QueryMind Research Lab

A research-grade search autocomplete engine that visualizes 9 deep learning and data science concepts through an interactive dark-themed dashboard.

## How to Run

```bash
python app.py
```

The app runs on port 5000 by default.

To train the fine-tuned GPT-2 model first (optional — base GPT-2 is used as fallback):
```bash
python model/train.py
```

## Architecture

**Backend (Flask):**
- `app.py` — main Flask server with all API routes
- `model/trie.py` — Trie prefix tree for O(k) autocomplete
- `model/markov.py` — N-gram Markov chain language model
- `model/gbdt.py` — Gradient-Boosted Decision Trees (sklearn)
- `model/embeddings.py` — GPT-2 embedding extraction, PCA projection, LSTM gate simulation, gradient descent surface
- `model/inference.py` — GPT-2 transformer inference + attention extraction
- `model/session_store.py` — in-memory user session tracking

**Frontend (Vanilla JS + D3.js + Three.js + Chart.js):**
- `templates/index.html` — 10-tab research dashboard
- `static/style.css` — dark research theme
- `static/main.js` — tab switching, search engine, pipeline visualization
- `static/viz_trie.js` — D3.js radial trie explorer
- `static/viz_markov.js` — D3.js force-directed Markov state graph
- `static/viz_attention.js` — Canvas attention heatmap (Viridis colormap)
- `static/viz_lstm.js` — SVG LSTM cell architecture + Chart.js timeline
- `static/viz_embeddings.js` — Three.js 3D word embedding point cloud
- `static/viz_gradient.js` — Three.js 3D loss surface + gradient descent animation
- `static/viz_gbdt.js` — D3.js decision tree + feature importance charts
- `static/viz_history.js` — session analytics, word cloud, intent distribution
- `static/viz_metrics.js` — TensorBoard-style training metrics

## Research Modules

| Tab | Concept | Key Tech |
|-----|---------|----------|
| Search Engine | Multi-model autocomplete | GPT-2 + Trie + Markov merged |
| Trie Explorer | O(k) prefix matching | D3.js radial tree |
| Markov Chain | N-gram state transitions | D3.js force graph |
| Attention Maps | Self-attention visualization | Canvas heatmap (Viridis) |
| LSTM Internals | Gate activations per character | SVG architecture + Chart.js |
| 3D Embeddings | PCA-projected word vectors | Three.js point cloud |
| Gradient Descent | Rosenbrock loss surface | Three.js 3D mesh + animation |
| Decision Trees | GBDT tree #1 + feature importance | D3.js tree + Chart.js |
| User History | Session analytics, word cloud | Chart.js doughnut + scatter |
| Metrics | Training loss + perplexity | Chart.js line charts |

## API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/api/search` | POST | All-in-one: Trie + Markov + Transformer |
| `/api/predict` | POST | GPT-2 transformer autocomplete |
| `/api/trie?prefix=` | GET | Trie search + structure |
| `/api/markov?query=&n=` | GET | Markov predictions + force graph |
| `/api/attention?query=&layer=` | GET | Multi-head attention weights |
| `/api/lstm?query=` | GET | LSTM gate activations |
| `/api/embeddings?words=` | GET | 3D PCA word vectors |
| `/api/gradient` | GET | Loss surface + descent path |
| `/api/gbdt?query=` | GET | GBDT tree + feature importance |
| `/api/history` | GET | User session history |
| `/api/metrics` | GET | Training metrics |

## User Preferences

- Pure black theme — no gradients, no glows, no rounded corners (border-radius: 0 everywhere)
- No emoji icons anywhere in the UI — plain text labels only
- Minimal, human-written aesthetic: flat boxes, sharp edges, muted color palette
- All visualizations should work without the fine-tuned model (base GPT-2 used as fallback)
- Keep the existing Flask + Vanilla JS stack — no frontend framework migration
