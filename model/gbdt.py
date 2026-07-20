"""
Gradient-Boosted Decision Trees for Query Ranking
==================================================
GBDT combines many shallow decision trees in sequence, each one
correcting the residual error of the previous. Used by Bing, Yandex,
and many search engines for learning-to-rank (LTR) tasks.

Formula:  F_m(x) = F_{m-1}(x) + η · h_m(x)
Where h_m is a new tree fitted to the negative gradient of the loss.
"""

import os
import numpy as np
from collections import defaultdict

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ── Vocabulary from corpus ──────────────────────────────────────────────────
def _load_corpus():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'queries.txt')
    with open(data_path, 'r') as f:
        return [l.strip().lower() for l in f if l.strip()]


def _build_vocab_freq(corpus):
    freq = defaultdict(int)
    for q in corpus:
        for w in q.split():
            freq[w] += 1
    return freq


# ── Feature extraction ───────────────────────────────────────────────────────
FEATURE_NAMES = [
    'query_length', 'word_count', 'starts_how', 'starts_what',
    'starts_best', 'starts_i_want', 'starts_why', 'contains_learn',
    'contains_build', 'contains_data', 'contains_machine', 'contains_python',
    'last_word_freq', 'avg_word_freq', 'unique_words', 'char_entropy'
]


def _char_entropy(text: str) -> float:
    from collections import Counter
    import math
    counts = Counter(text)
    total = len(text) or 1
    return -sum((c / total) * math.log(c / total + 1e-10) for c in counts.values())


def extract_features(query: str, vocab_freq: dict) -> np.ndarray:
    q = query.strip().lower()
    words = q.split()
    if not words:
        return np.zeros(len(FEATURE_NAMES))

    last_word_freq = vocab_freq.get(words[-1], 0)
    avg_freq = np.mean([vocab_freq.get(w, 0) for w in words])

    feats = [
        len(q),                                        # query_length
        len(words),                                    # word_count
        1 if q.startswith('how') else 0,               # starts_how
        1 if q.startswith('what') else 0,              # starts_what
        1 if q.startswith('best') else 0,              # starts_best
        1 if q.startswith('i want') else 0,            # starts_i_want
        1 if q.startswith('why') else 0,               # starts_why
        1 if 'learn' in q else 0,                      # contains_learn
        1 if 'build' in q else 0,                      # contains_build
        1 if 'data' in q else 0,                       # contains_data
        1 if 'machine' in q else 0,                    # contains_machine
        1 if 'python' in q else 0,                     # contains_python
        last_word_freq,                                # last_word_freq
        avg_freq,                                      # avg_word_freq
        len(set(words)),                               # unique_words
        _char_entropy(q),                              # char_entropy
    ]
    return np.array(feats, dtype=float)


# ── GBDT Model ──────────────────────────────────────────────────────────────
class GBDTRanker:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_OK else None
        self.vocab_freq = {}
        self.corpus = []
        self.feature_importances_ = None
        self.tree_structure_ = None
        self._trained = False

    def train(self):
        if not SKLEARN_OK:
            return False

        self.corpus = _load_corpus()
        self.vocab_freq = _build_vocab_freq(self.corpus)

        # Build training data
        # Label: query is "popular" if its words are frequently searched
        X, y = [], []
        freq_values = [_total_freq(q, self.vocab_freq) for q in self.corpus]
        median_freq = np.median(freq_values) if freq_values else 1

        for query in self.corpus:
            feats = extract_features(query, self.vocab_freq)
            label = 1 if _total_freq(query, self.vocab_freq) >= median_freq else 0
            X.append(feats)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Normalize
        X_scaled = self.scaler.fit_transform(X)

        self.model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        self.feature_importances_ = self.model.feature_importances_.tolist()

        # Extract first tree for visualization
        self.tree_structure_ = _tree_to_dict(
            self.model.estimators_[0][0],
            FEATURE_NAMES
        )
        self._trained = True
        return True

    def rank(self, suggestions: list, query: str) -> list:
        """Re-rank autocomplete suggestions using GBDT."""
        if not self._trained or not SKLEARN_OK:
            return suggestions

        for s in suggestions:
            feats = extract_features(s.get('text', query), self.vocab_freq)
            feats_scaled = self.scaler.transform([feats])
            proba = self.model.predict_proba(feats_scaled)[0][1]
            s['gbdt_score'] = round(float(proba), 4)
            s['gbdt_rank_boost'] = round(float(proba * 0.3), 4)

        suggestions.sort(key=lambda x: -(x.get('confidence', 0) + x.get('gbdt_rank_boost', 0)))
        return suggestions

    def explain(self, query: str):
        """Feature importance explanation for a specific query."""
        if not self._trained or not SKLEARN_OK:
            return {}

        feats = extract_features(query, self.vocab_freq)
        feats_scaled = self.scaler.transform([feats])
        proba = self.model.predict_proba(feats_scaled)[0][1]

        feature_contrib = [
            {
                'feature': FEATURE_NAMES[i],
                'value': round(float(feats[i]), 4),
                'importance': round(float(self.feature_importances_[i]), 4)
            }
            for i in range(len(FEATURE_NAMES))
        ]
        feature_contrib.sort(key=lambda x: -x['importance'])

        return {
            'query': query,
            'popularity_score': round(float(proba), 4),
            'features': feature_contrib,
            'global_importance': [
                {'feature': FEATURE_NAMES[i], 'importance': round(float(v), 4)}
                for i, v in enumerate(self.feature_importances_)
            ],
            'tree': self.tree_structure_
        }

    def get_boosting_steps(self):
        """Return per-tree loss reduction for the boosting curve visualization."""
        if not self._trained or not SKLEARN_OK:
            return []
        stages = self.model.train_score_.tolist()
        return [{'step': i + 1, 'loss': round(v, 6)} for i, v in enumerate(stages)]


def _total_freq(query: str, vocab_freq: dict) -> float:
    words = query.split()
    return sum(vocab_freq.get(w, 0) for w in words) / max(len(words), 1)


def _tree_to_dict(tree, feature_names: list, node_id: int = 0, depth: int = 0, max_depth: int = 3):
    """Recursively convert sklearn tree structure to JSON-serializable dict."""
    sk_tree = tree.tree_
    left = sk_tree.children_left[node_id]
    right = sk_tree.children_right[node_id]

    if depth > max_depth or left == -1:  # leaf
        return {
            'type': 'leaf',
            'value': round(float(sk_tree.value[node_id][0][0]), 6),
            'samples': int(sk_tree.n_node_samples[node_id]),
            'impurity': round(float(sk_tree.impurity[node_id]), 4),
            'depth': depth
        }

    feat_idx = sk_tree.feature[node_id]
    feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'f{feat_idx}'

    return {
        'type': 'split',
        'feature': feat_name,
        'threshold': round(float(sk_tree.threshold[node_id]), 4),
        'samples': int(sk_tree.n_node_samples[node_id]),
        'impurity': round(float(sk_tree.impurity[node_id]), 4),
        'depth': depth,
        'left': _tree_to_dict(tree, feature_names, left, depth + 1, max_depth),
        'right': _tree_to_dict(tree, feature_names, right, depth + 1, max_depth)
    }


# ── Singleton ────────────────────────────────────────────────────────────────
_ranker = None


def get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = GBDTRanker()
        _ranker.train()
    return _ranker
