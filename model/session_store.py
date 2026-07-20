"""
User Session & History Tracking
================================
Tracks per-session query history for personalized autocomplete.
In production this would use Redis or a persistent DB.
"""

import time
from collections import defaultdict, Counter


class SessionStore:
    def __init__(self):
        self._sessions = {}   # session_id -> session_data

    def _init_session(self, session_id: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                'id': session_id,
                'queries': [],
                'word_counts': Counter(),
                'intent_counts': Counter(),
                'session_start': time.time(),
                'last_active': time.time(),
            }
        return self._sessions[session_id]

    def add_query(self, session_id: str, query: str):
        sess = self._init_session(session_id)
        ts = time.time()

        # Detect query intent
        intent = _classify_intent(query)

        sess['queries'].append({
            'text': query,
            'timestamp': ts,
            'time_ago': 0,
            'intent': intent,
            'length': len(query),
            'word_count': len(query.split())
        })
        sess['last_active'] = ts

        for w in query.lower().split():
            sess['word_counts'][w] += 1

        sess['intent_counts'][intent] += 1

        # Keep only last 100 queries
        if len(sess['queries']) > 100:
            sess['queries'] = sess['queries'][-100:]

    def get_history(self, session_id: str):
        sess = self._init_session(session_id)
        now = time.time()

        # Compute time_ago for each query
        queries = []
        for q in reversed(sess['queries'][-20:]):
            ago = now - q['timestamp']
            q_copy = dict(q)
            q_copy['time_ago'] = _format_ago(ago)
            queries.append(q_copy)

        # Top words (excluding stopwords)
        stopwords = {'to', 'a', 'the', 'of', 'in', 'is', 'it', 'i', 'and', 'for', 'how',
                     'what', 'want', 'have', 'my', 'at', 'on', 'be', 'or', 'that', 'with'}
        top_words = [
            {'word': w, 'count': c}
            for w, c in sess['word_counts'].most_common(20)
            if w not in stopwords
        ]

        # Session duration
        duration = now - sess['session_start']

        # Query length histogram (bins: 0-20, 21-40, 41-60, 61+)
        lengths = [q['length'] for q in sess['queries']]
        length_hist = [
            {'bin': '0–20', 'count': sum(1 for l in lengths if l <= 20)},
            {'bin': '21–40', 'count': sum(1 for l in lengths if 21 <= l <= 40)},
            {'bin': '41–60', 'count': sum(1 for l in lengths if 41 <= l <= 60)},
            {'bin': '61+', 'count': sum(1 for l in lengths if l > 60)},
        ]

        # Query frequency over time (last 20 queries relative timing)
        timing = []
        if sess['queries']:
            base_t = sess['queries'][0]['timestamp']
            for q in sess['queries'][-20:]:
                timing.append({
                    'text': q['text'][:30],
                    'elapsed': round(q['timestamp'] - base_t, 1)
                })

        return {
            'recent_queries': queries,
            'total_queries': len(sess['queries']),
            'unique_words': len(sess['word_counts']),
            'top_words': top_words[:10],
            'intent_distribution': dict(sess['intent_counts']),
            'session_duration': round(duration, 1),
            'length_histogram': length_hist,
            'query_timing': timing,
            'avg_query_length': round(sum(lengths) / max(len(lengths), 1), 1)
        }

    def get_personalized_boost(self, session_id: str, suggestions: list) -> list:
        """Boost suggestions that match user's history patterns."""
        sess = self._sessions.get(session_id)
        if not sess:
            return suggestions

        for s in suggestions:
            boost = 0.0
            text = s.get('text', '').lower()
            for w in text.split():
                boost += sess['word_counts'].get(w, 0) * 0.01
            s['personal_boost'] = round(min(boost, 0.2), 4)
            s['confidence'] = round(s.get('confidence', 0) + boost * 0.05, 4)

        return suggestions


def _classify_intent(query: str) -> str:
    q = query.lower()
    if any(q.startswith(p) for p in ('how', 'why', 'when', 'where', 'what is')):
        return 'informational'
    if any(w in q for w in ('learn', 'study', 'understand', 'tutorial')):
        return 'educational'
    if any(w in q for w in ('build', 'create', 'make', 'develop', 'implement')):
        return 'project'
    if any(w in q for w in ('job', 'career', 'internship', 'salary', 'interview')):
        return 'career'
    if any(w in q for w in ('best', 'top', 'vs', 'compare', 'difference')):
        return 'comparative'
    return 'navigational'


def _format_ago(seconds: float) -> str:
    if seconds < 60:
        return f'{int(seconds)}s ago'
    if seconds < 3600:
        return f'{int(seconds / 60)}m ago'
    return f'{int(seconds / 3600)}h ago'


# ── Singleton ────────────────────────────────────────────────────────────────
_store = SessionStore()

def get_store() -> SessionStore:
    return _store
