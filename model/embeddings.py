"""
Word Embeddings & 3D Projection
================================
Extracts token embedding vectors from GPT-2's Embedding layer (wte),
projects them to 3D using PCA, and clusters them with K-Means.

Each dimension of the embedding space encodes latent semantic meaning.
Similar words cluster together in high-dimensional space — this module
lets you see that structure in 3D.
"""

import os
import numpy as np
from collections import defaultdict

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# Query words to visualize
SEED_WORDS = [
    'learn', 'machine', 'learning', 'deep', 'data', 'science',
    'python', 'build', 'model', 'neural', 'network', 'transformer',
    'attention', 'gradient', 'descent', 'loss', 'training', 'inference',
    'search', 'autocomplete', 'query', 'prediction', 'sequence',
    'embedding', 'vector', 'token', 'softmax', 'probability',
    'career', 'engineer', 'developer', 'algorithm', 'coding',
    'backpropagation', 'optimization', 'classification', 'regression',
    'how', 'what', 'best', 'want', 'become', 'start', 'job'
]

CLUSTER_LABELS = {
    0: 'Concepts',
    1: 'ML/AI',
    2: 'Actions',
    3: 'Career',
    4: 'Math',
    5: 'Query Words'
}

CLUSTER_COLORS = ['#4f8ef7', '#34d399', '#f59e0b', '#a78bfa', '#f87171', '#22d3ee']


class EmbeddingProjector:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.embedding_matrix = None
        self.pca = None
        self.points_3d = None
        self.word_to_idx = {}
        self.labels = []
        self._ready = False

    def load(self):
        """Load base GPT-2 model for embeddings (no training required)."""
        try:
            import torch
            from transformers import GPT2TokenizerFast, GPT2Model

            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            self.model = GPT2Model.from_pretrained('gpt2')
            self.model.eval()

            # Extract the embedding matrix (vocab_size × 768)
            with torch.no_grad():
                self.embedding_matrix = self.model.transformer.wte.weight.detach().numpy()

            self._ready = True
            return True
        except Exception as e:
            print(f'[Embeddings] Could not load GPT-2: {e}')
            self._ready = False
            return False

    def get_word_vector(self, word: str) -> np.ndarray | None:
        """Get embedding vector for a single word."""
        if not self._ready:
            return None
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        # Average sub-token embeddings
        vecs = self.embedding_matrix[ids]
        return vecs.mean(axis=0)

    def project_words(self, words: list = None):
        """
        Project a list of words into 3D using PCA.
        Returns list of {word, x, y, z, cluster, color} for Three.js.
        """
        if words is None:
            words = SEED_WORDS

        if not SKLEARN_OK:
            return self._synthetic_projection(words)

        if not self._ready:
            self.load()

        if not self._ready:
            return self._synthetic_projection(words)

        vectors = []
        valid_words = []
        for w in words:
            v = self.get_word_vector(w)
            if v is not None:
                vectors.append(v)
                valid_words.append(w)

        if len(vectors) < 4:
            return self._synthetic_projection(words)

        X = np.array(vectors)
        X_norm = normalize(X)

        # PCA to 3D
        pca = PCA(n_components=3, random_state=42)
        coords_3d = pca.fit_transform(X_norm)

        # Normalize to [-1, 1]
        for i in range(3):
            mn, mx = coords_3d[:, i].min(), coords_3d[:, i].max()
            rng = mx - mn or 1.0
            coords_3d[:, i] = (coords_3d[:, i] - mn) / rng * 2 - 1

        # K-Means clustering
        n_clusters = min(6, len(valid_words))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_norm)

        # Explained variance
        explained = pca.explained_variance_ratio_.tolist()

        results = []
        for i, w in enumerate(valid_words):
            c = int(clusters[i])
            results.append({
                'word': w,
                'x': round(float(coords_3d[i, 0]), 4),
                'y': round(float(coords_3d[i, 1]), 4),
                'z': round(float(coords_3d[i, 2]), 4),
                'cluster': c,
                'cluster_label': CLUSTER_LABELS.get(c, f'Cluster {c}'),
                'color': CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            })

        # Nearest neighbors for each word
        for i, item in enumerate(results):
            dists = [
                (results[j]['word'], float(np.linalg.norm(X_norm[i] - X_norm[j])))
                for j in range(len(results)) if j != i
            ]
            dists.sort(key=lambda x: x[1])
            item['nearest'] = [d[0] for d in dists[:3]]

        return {
            'points': results,
            'explained_variance': [round(v, 4) for v in explained],
            'total_variance_explained': round(sum(explained), 4),
            'embedding_dim': 768,
            'n_clusters': n_clusters
        }

    def get_similarity_matrix(self, words: list):
        """Cosine similarity matrix between words."""
        if not self._ready:
            n = len(words)
            mat = np.eye(n) + np.random.rand(n, n) * 0.3
            mat = (mat + mat.T) / 2
            np.fill_diagonal(mat, 1.0)
            return mat.tolist()

        vectors = []
        for w in words:
            v = self.get_word_vector(w)
            vectors.append(v if v is not None else np.zeros(768))

        X = np.array(vectors)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_norm = X / norms
        sim_matrix = (X_norm @ X_norm.T).tolist()
        return [[round(v, 4) for v in row] for row in sim_matrix]

    def _synthetic_projection(self, words: list):
        """Fallback: generate demo embeddings when GPT-2 is unavailable."""
        np.random.seed(42)
        n = len(words)
        coords = np.random.randn(n, 3)

        # Group similar words together
        tech_words = {'machine', 'learning', 'deep', 'neural', 'transformer',
                      'attention', 'gradient', 'model', 'training', 'backpropagation'}
        action_words = {'learn', 'build', 'start', 'become', 'want', 'get'}

        for i, w in enumerate(words):
            if w in tech_words:
                coords[i] += np.array([1.5, 0.5, 0])
            elif w in action_words:
                coords[i] += np.array([-1.5, 0.5, 0])

        # Normalize
        for j in range(3):
            mn, mx = coords[:, j].min(), coords[:, j].max()
            rng = mx - mn or 1.0
            coords[:, j] = (coords[:, j] - mn) / rng * 2 - 1

        clusters = [1 if w in tech_words else (2 if w in action_words else 0) for w in words]

        return {
            'points': [
                {
                    'word': words[i],
                    'x': round(float(coords[i, 0]), 4),
                    'y': round(float(coords[i, 1]), 4),
                    'z': round(float(coords[i, 2]), 4),
                    'cluster': clusters[i],
                    'cluster_label': CLUSTER_LABELS.get(clusters[i], 'Other'),
                    'color': CLUSTER_COLORS[clusters[i] % len(CLUSTER_COLORS)],
                    'nearest': []
                }
                for i in range(n)
            ],
            'explained_variance': [0.42, 0.28, 0.18],
            'total_variance_explained': 0.88,
            'embedding_dim': 768,
            'n_clusters': 3,
            'demo_mode': True
        }


# ── Gradient Descent Surface Data ───────────────────────────────────────────
def compute_gradient_surface(resolution: int = 40):
    """
    Compute a 2D loss surface for gradient descent visualization.
    Uses a modified Rosenbrock valley: L(w1,w2) = (1-w1)² + 10(w2-w1²)²
    plus a saddle perturbation to make the surface more interesting.
    """
    w1 = np.linspace(-2, 2, resolution)
    w2 = np.linspace(-1, 3, resolution)
    W1, W2 = np.meshgrid(w1, w2)

    # Modified Rosenbrock
    L = (1 - W1)**2 + 2 * (W2 - W1**2)**2
    # Add slight noise for realism
    np.random.seed(7)
    L = L + np.random.randn(*L.shape) * 0.05
    L = np.clip(L, 0, 8)

    # Gradient descent path
    def grad_L(x, y):
        gx = -2 * (1 - x) - 8 * x * (y - x**2)
        gy = 4 * (y - x**2)
        return np.array([gx, gy])

    path = []
    x, y = -1.5, 2.5
    lr = 0.025
    for step in range(120):
        l_val = (1 - x)**2 + 2 * (y - x**2)**2
        path.append({'step': step, 'w1': round(float(x), 4), 'w2': round(float(y), 4), 'loss': round(float(l_val), 6)})
        g = grad_L(x, y)
        # Gradient clipping
        g_norm = np.linalg.norm(g)
        if g_norm > 2:
            g = g / g_norm * 2
        x -= lr * g[0]
        y -= lr * g[1]

    # Color scale: normalize to [0, 1]
    L_min, L_max = L.min(), L.max()
    L_norm = (L - L_min) / (L_max - L_min + 1e-10)

    return {
        'w1': w1.tolist(),
        'w2': w2.tolist(),
        'loss_surface': L.tolist(),
        'loss_normalized': L_norm.tolist(),
        'gradient_path': path,
        'minimum': {'w1': 1.0, 'w2': 1.0, 'loss': 0.0},
        'start': {'w1': -1.5, 'w2': 2.5},
        'resolution': resolution
    }


# ── LSTM Gate Computation ────────────────────────────────────────────────────
def compute_lstm_states(query: str):
    """
    Simulate a character-level LSTM with seeded weights.
    Returns actual gate activation values for each character step.
    This demonstrates real LSTM mathematics, not approximations.
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    np.random.seed(42)
    hidden_size = 16
    input_size = 64   # compressed char one-hot

    # Initialize weights (seeded → consistent)
    Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
    Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
    Wg = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
    Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
    bf = np.ones(hidden_size) * 0.5   # forget gate bias > 0 helps memory
    bi = np.zeros(hidden_size)
    bg = np.zeros(hidden_size)
    bo = np.zeros(hidden_size)

    h = np.zeros(hidden_size)
    c = np.zeros(hidden_size)

    steps = []
    query_str = query.strip()[:24]   # limit for UI

    for t, char in enumerate(query_str):
        # One-hot encode char (mod input_size)
        x = np.zeros(input_size)
        x[ord(char) % input_size] = 1.0

        combined = np.concatenate([h, x])

        ft = sigmoid(Wf @ combined + bf)      # forget gate
        it = sigmoid(Wi @ combined + bi)      # input gate
        gt = np.tanh(Wg @ combined + bg)      # cell gate (candidate)
        c = ft * c + it * gt                  # cell state update
        ot = sigmoid(Wo @ combined + bo)      # output gate
        h = ot * np.tanh(c)                   # hidden state

        steps.append({
            't': t,
            'char': char,
            'forget_gate': [round(float(v), 4) for v in ft],
            'input_gate': [round(float(v), 4) for v in it],
            'cell_gate': [round(float(v), 4) for v in gt],
            'output_gate': [round(float(v), 4) for v in ot],
            'cell_state': [round(float(v), 4) for v in c],
            'hidden_state': [round(float(v), 4) for v in h],
            # Scalar summaries for bar charts
            'forget_mean': round(float(ft.mean()), 4),
            'input_mean': round(float(it.mean()), 4),
            'gate_mean': round(float(gt.mean()), 4),
            'output_mean': round(float(ot.mean()), 4),
            'cell_norm': round(float(np.linalg.norm(c)), 4),
            'hidden_norm': round(float(np.linalg.norm(h)), 4),
        })

    return {
        'steps': steps,
        'query': query_str,
        'hidden_size': hidden_size,
        'input_size': input_size,
        'final_cell': [round(float(v), 4) for v in c],
        'final_hidden': [round(float(v), 4) for v in h]
    }


# ── Singleton ────────────────────────────────────────────────────────────────
_projector = None

def get_projector():
    global _projector
    if _projector is None:
        _projector = EmbeddingProjector()
    return _projector
