"""
QueryMind Research Lab — Flask Application
==========================================
Research-grade search autocomplete with 9 visualization modules:
Transformer · Markov Chain · Trie · Attention · LSTM · 3D Embeddings ·
Gradient Descent · GBDT · User History
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'querymind-dev-key-2024')
CORS(app)

# ── Lazy-load heavy modules ───────────────────────────────────────────────────
_inference_loaded = False
_trie = None
_markov2 = None
_markov3 = None
_ranker = None
_projector = None


def _get_inference():
    from model.inference import autocomplete as _ac, get_attention_weights, get_metrics
    return _ac, get_attention_weights, get_metrics


def _get_trie():
    global _trie
    if _trie is None:
        from model.trie import get_trie
        _trie = get_trie()
    return _trie


def _get_markov(n=2):
    global _markov2, _markov3
    from model.markov import get_markov
    if n == 2:
        if _markov2 is None:
            _markov2 = get_markov(2)
        return _markov2
    if _markov3 is None:
        _markov3 = get_markov(3)
    return _markov3


def _get_ranker():
    global _ranker
    if _ranker is None:
        from model.gbdt import get_ranker
        _ranker = get_ranker()
    return _ranker


def _get_projector():
    global _projector
    if _projector is None:
        from model.embeddings import get_projector
        _projector = get_projector()
        _projector.load()
    return _projector


def _session_id():
    if 'uid' not in session:
        session['uid'] = str(uuid.uuid4())[:8]
    return session['uid']


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


# ── API: Transformer Autocomplete ─────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        query = request.json.get('query', '').strip()
        if not query:
            return jsonify({'error': 'empty query'}), 400

        ac_fn, _, _ = _get_inference()
        result = ac_fn(query, k=8)

        # Log to session history
        from model.session_store import get_store
        get_store().add_query(_session_id(), query)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'suggestions': [], 'tokens': []}), 500


# ── API: Trie ─────────────────────────────────────────────────────────────────
@app.route('/api/trie')
def trie_api():
    prefix = request.args.get('prefix', '').strip()
    trie = _get_trie()

    result = trie.search_prefix(prefix, max_results=10)
    result['stats'] = trie.get_stats()
    result['structure'] = trie.get_structure(max_depth=4, max_children=5)
    return jsonify(result)


# ── API: Markov Chain ─────────────────────────────────────────────────────────
@app.route('/api/markov')
def markov_api():
    query = request.args.get('query', 'how to learn').strip()
    n = int(request.args.get('n', 2))
    n = max(2, min(n, 3))

    mc = _get_markov(n)
    predictions = mc.predict(query, top_k=8)
    completions = mc.generate_completions(query, num_completions=6)
    graph = mc.get_transition_graph(query, depth=2)

    # Build transition matrix for top query words
    words = query.split()[-4:] if query else []
    if len(words) >= 2:
        matrix = mc.get_transition_matrix(words)
    else:
        matrix = []

    return jsonify({
        'query': query,
        'n': n,
        'next_word_predictions': predictions,
        'completions': completions,
        'transition_graph': graph,
        'transition_matrix': {'words': words, 'matrix': matrix},
        'stats': mc.get_stats()
    })


# ── API: Attention Weights ────────────────────────────────────────────────────
@app.route('/api/attention')
def attention_api():
    query = request.args.get('query', 'how to learn machine learning').strip()
    layer = int(request.args.get('layer', 5))
    try:
        _, get_attn, _ = _get_inference()
        result = get_attn(query, layer=layer)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── API: LSTM Gate States ─────────────────────────────────────────────────────
@app.route('/api/lstm')
def lstm_api():
    query = request.args.get('query', 'how to learn machine').strip()
    from model.embeddings import compute_lstm_states
    result = compute_lstm_states(query)
    return jsonify(result)


# ── API: 3D Word Embeddings ───────────────────────────────────────────────────
@app.route('/api/embeddings')
def embeddings_api():
    words_param = request.args.get('words', '')
    from model.embeddings import SEED_WORDS, get_projector
    proj = _get_projector()
    if words_param:
        words = [w.strip() for w in words_param.split(',') if w.strip()][:40]
    else:
        words = SEED_WORDS
    result = proj.project_words(words)
    return jsonify(result)


@app.route('/api/similarity')
def similarity_api():
    words_param = request.args.get('words', 'learn,machine,data,build,career')
    proj = _get_projector()
    words = [w.strip() for w in words_param.split(',') if w.strip()][:12]
    matrix = proj.get_similarity_matrix(words)
    return jsonify({'words': words, 'matrix': matrix})


# ── API: Gradient Descent Surface ────────────────────────────────────────────
@app.route('/api/gradient')
def gradient_api():
    from model.embeddings import compute_gradient_surface
    result = compute_gradient_surface(resolution=35)
    return jsonify(result)


# ── API: GBDT ────────────────────────────────────────────────────────────────
@app.route('/api/gbdt')
def gbdt_api():
    query = request.args.get('query', 'how to learn machine learning').strip()
    ranker = _get_ranker()
    explanation = ranker.explain(query)
    boosting_steps = ranker.get_boosting_steps()
    explanation['boosting_curve'] = boosting_steps[:30]
    return jsonify(explanation)


# ── API: User History ─────────────────────────────────────────────────────────
@app.route('/api/history')
def history_api():
    from model.session_store import get_store
    return jsonify(get_store().get_history(_session_id()))


@app.route('/api/history/add', methods=['POST'])
def history_add():
    query = request.json.get('query', '').strip()
    if query:
        from model.session_store import get_store
        get_store().add_query(_session_id(), query)
    return jsonify({'ok': True})


# ── API: Training Metrics ─────────────────────────────────────────────────────
@app.route('/api/metrics')
def metrics_api():
    try:
        _, _, get_metrics = _get_inference()
        return jsonify(get_metrics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── API: All-in-one search (combines all models) ──────────────────────────────
@app.route('/api/search', methods=['POST'])
def search_all():
    """Aggregate results from Transformer + Markov + Trie in one call."""
    query = request.json.get('query', '').strip()
    if not query:
        return jsonify({'error': 'empty query'}), 400

    results = {}

    # Trie (fastest — always available)
    try:
        trie = _get_trie()
        trie_res = trie.search_prefix(query, max_results=6)
        results['trie'] = trie_res.get('suggestions', [])
    except Exception as e:
        results['trie'] = []
        results['trie_error'] = str(e)

    # Markov chain
    try:
        mc = _get_markov(2)
        completions = mc.generate_completions(query, num_completions=6)
        results['markov'] = completions
    except Exception as e:
        results['markov'] = []
        results['markov_error'] = str(e)

    # Transformer (slowest — may fail if model not trained)
    try:
        ac_fn, _, _ = _get_inference()
        tf_res = ac_fn(query, k=6)
        results['transformer'] = tf_res.get('suggestions', [])
        results['tokens'] = tf_res.get('tokens', [])
        results['is_finetuned'] = tf_res.get('is_finetuned', False)
    except Exception as e:
        results['transformer'] = []
        results['transformer_error'] = str(e)

    # Session history
    from model.session_store import get_store
    get_store().add_query(_session_id(), query)

    return jsonify(results)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
