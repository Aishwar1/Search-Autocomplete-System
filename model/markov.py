"""
Markov Chain Language Model for Query Prediction
=================================================
An N-gram Markov chain models the probability of the next word
given the previous N-1 words. P(w_n | w_{n-N+1}, ..., w_{n-1})

This is the backbone of early search engines like Yahoo! and early Google.
"""

import os
import math
import random
from collections import defaultdict


class MarkovChain:
    def __init__(self, n: int = 2):
        """
        n=1: unigram (word frequencies)
        n=2: bigram (P(word | prev_word))
        n=3: trigram (P(word | prev_2_words))
        """
        self.n = n
        # transitions[(context_tuple)] = {next_word: count}
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.start_counts = defaultdict(int)   # first words
        self.word_counts = defaultdict(int)    # unigram counts
        self.total_tokens = 0
        self.vocabulary = set()
        self.corpus = []

    def train(self, corpus: list):
        """Train on a list of query strings."""
        self.corpus = [q.strip().lower() for q in corpus if q.strip()]

        for query in self.corpus:
            words = query.split()
            if not words:
                continue

            # Track unigrams
            for w in words:
                self.word_counts[w] += 1
                self.vocabulary.add(w)
                self.total_tokens += 1

            # Track start word
            self.start_counts[words[0]] += 1

            # Build n-gram transitions
            # Add padding for context shorter than n-1
            padded = ['<START>'] * (self.n - 1) + words + ['<END>']
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i: i + self.n - 1])
                next_word = padded[i + self.n - 1]
                self.transitions[context][next_word] += 1

    def predict(self, query: str, top_k: int = 8):
        """
        Given a partial query, predict the next most likely word(s).
        Returns list of (word, probability) tuples sorted by probability.
        """
        query = query.strip().lower()
        words = query.split() if query else []

        # Build context from last n-1 words
        if len(words) >= self.n - 1:
            context = tuple(words[-(self.n - 1):]) if self.n > 1 else tuple()
        else:
            # Pad with START tokens
            padding = ['<START>'] * (self.n - 1 - len(words))
            context = tuple(padding + words)

        candidates = self.transitions.get(context, {})

        # Fallback: try shorter context (backoff)
        if not candidates and len(words) > 0:
            context_1 = (words[-1],)
            candidates = self.transitions.get(context_1, {})

        # Fallback: use unigram probabilities
        if not candidates:
            candidates = dict(self.word_counts)

        # Remove special tokens
        candidates = {k: v for k, v in candidates.items()
                      if k not in ('<START>', '<END>')}

        if not candidates:
            return []

        total = sum(candidates.values())
        predictions = [
            {
                'word': word,
                'count': count,
                'probability': count / total,
                'log_prob': math.log(count / total + 1e-10)
            }
            for word, count in candidates.items()
        ]
        predictions.sort(key=lambda x: -x['probability'])
        return predictions[:top_k]

    def generate_completions(self, query: str, num_completions: int = 6, max_words: int = 4):
        """Generate full query completions using the Markov chain."""
        completions = []
        seen = set()

        for _ in range(num_completions * 5):
            if len(completions) >= num_completions:
                break

            current = query.strip().lower()
            words = current.split()
            score = 1.0

            for _ in range(max_words):
                preds = self.predict(current, top_k=10)
                if not preds:
                    break

                # Sample from top predictions (temperature sampling)
                weights = [p['probability'] ** 1.5 for p in preds[:6]]
                total_w = sum(weights)
                if total_w == 0:
                    break
                probs = [w / total_w for w in weights]

                r = random.random()
                cumulative = 0.0
                chosen = preds[0]
                for pred, prob in zip(preds[:6], probs):
                    cumulative += prob
                    if r <= cumulative:
                        chosen = pred
                        break

                next_word = chosen['word']
                if next_word == '<END>':
                    break

                score *= chosen['probability']
                current = current + ' ' + next_word
                words.append(next_word)

            if current not in seen and current != query.strip().lower():
                seen.add(current)
                completions.append({
                    'text': current,
                    'confidence': min(score * 10, 0.99),
                    'model': 'markov'
                })

        return completions

    def get_transition_graph(self, query: str, depth: int = 2):
        """
        Build a force-directed graph representation of Markov transitions
        centered on the words in the query. Used for D3.js visualization.
        """
        query = query.strip().lower()
        words = query.split() if query else ['how', 'to', 'learn']

        nodes = {}  # id -> node_data
        edges = []
        visited = set()

        node_id = [0]

        def get_node(word):
            if word not in nodes:
                freq = self.word_counts.get(word, 1)
                nodes[word] = {
                    'id': word,
                    'label': word,
                    'frequency': freq,
                    'size': max(8, min(30, freq * 2)),
                    'group': _word_group(word)
                }
            return nodes[word]

        def expand(word, current_depth):
            if current_depth > depth or word in visited:
                return
            visited.add(word)
            get_node(word)

            context = (word,)
            candidates = self.transitions.get(context, {})
            # top transitions only
            top = sorted(candidates.items(), key=lambda x: -x[1])[:5]
            total = sum(candidates.values()) or 1

            for next_word, count in top:
                if next_word in ('<START>', '<END>'):
                    continue
                get_node(next_word)
                prob = count / total
                edges.append({
                    'source': word,
                    'target': next_word,
                    'weight': prob,
                    'count': count,
                    'label': f'{prob:.2f}'
                })
                expand(next_word, current_depth + 1)

        for w in words[-3:]:
            expand(w, 0)

        return {
            'nodes': list(nodes.values()),
            'edges': edges,
            'query_words': words
        }

    def get_transition_matrix(self, words: list):
        """Return NxN transition probability matrix for heatmap visualization."""
        n = len(words)
        matrix = [[0.0] * n for _ in range(n)]

        for i, w1 in enumerate(words):
            context = (w1,)
            candidates = self.transitions.get(context, {})
            total = sum(candidates.values()) or 1
            for j, w2 in enumerate(words):
                matrix[i][j] = candidates.get(w2, 0) / total

        return matrix

    def get_stats(self):
        return {
            'n': self.n,
            'vocabulary_size': len(self.vocabulary),
            'total_tokens': self.total_tokens,
            'unique_contexts': len(self.transitions),
            'corpus_size': len(self.corpus)
        }


def _word_group(word: str) -> int:
    """Assign semantic group for color coding."""
    tech = {'python', 'java', 'data', 'machine', 'learning', 'ai', 'deep',
            'neural', 'model', 'algorithm', 'code', 'programming', 'software',
            'engineering', 'transformer', 'attention', 'lstm', 'gradient'}
    action = {'learn', 'build', 'start', 'get', 'become', 'work', 'prepare',
              'study', 'practice', 'improve', 'develop', 'create', 'design'}
    question = {'how', 'what', 'why', 'when', 'where', 'which', 'is', 'best'}
    goal = {'job', 'career', 'internship', 'startup', 'company', 'engineer',
            'scientist', 'developer', 'analyst'}

    if word in tech:
        return 1
    if word in action:
        return 2
    if word in question:
        return 3
    if word in goal:
        return 4
    return 0


# ---------- Module-level singletons ----------
_bigram = None
_trigram = None


def get_markov(n: int = 2):
    global _bigram, _trigram
    if n == 2 and _bigram:
        return _bigram
    if n == 3 and _trigram:
        return _trigram

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'queries.txt')
    with open(data_path, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]

    mc = MarkovChain(n=n)
    mc.train(corpus)

    if n == 2:
        _bigram = mc
    else:
        _trigram = mc
    return mc
