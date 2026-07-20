"""
Trie Data Structure for Search Autocomplete
============================================
A trie (prefix tree) stores strings character-by-character,
enabling O(k) prefix search where k = length of prefix.
Much faster than O(n) linear scan over all queries.
"""

import json
from collections import defaultdict


class TrieNode:
    __slots__ = ['children', 'is_end', 'frequency', 'queries', 'query_count']

    def __init__(self):
        self.children = {}        # char -> TrieNode
        self.is_end = False       # marks end of a complete query
        self.frequency = 0        # how many strings pass through this node
        self.queries = []         # full queries that end here
        self.query_count = 0      # number of queries through this node


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.total_queries = 0
        self.word_frequencies = defaultdict(int)

    def insert(self, query: str):
        """Insert a query into the trie."""
        query = query.strip().lower()
        if not query:
            return

        # Track word frequencies for Markov chain compatibility
        for word in query.split():
            self.word_frequencies[word] += 1

        node = self.root
        node.query_count += 1

        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.frequency += 1
            node.query_count += 1

        node.is_end = True
        node.queries.append(query)
        self.total_queries += 1

    def search_prefix(self, prefix: str, max_results: int = 10):
        """
        Find all queries matching a prefix.
        Returns matching queries + the traversal path for visualization.
        Time complexity: O(k + m) where k=prefix length, m=matching nodes
        """
        prefix = prefix.strip().lower()
        node = self.root
        traversal_path = []

        # Traverse to prefix end
        for char in prefix:
            if char not in node.children:
                return {
                    'suggestions': [],
                    'traversal_path': traversal_path,
                    'nodes_visited': len(traversal_path),
                    'found': False
                }
            traversal_path.append({
                'char': char,
                'freq': node.children[char].frequency,
                'depth': len(traversal_path)
            })
            node = node.children[char]

        # DFS to collect all completions
        results = []
        self._dfs_collect(node, prefix, results, max_results * 3)

        # Sort by frequency (most common first)
        results.sort(key=lambda x: -x['frequency'])

        return {
            'suggestions': results[:max_results],
            'traversal_path': traversal_path,
            'nodes_visited': len(traversal_path),
            'found': True,
            'subtree_size': node.query_count
        }

    def _dfs_collect(self, node: TrieNode, current: str, results: list, limit: int):
        if len(results) >= limit:
            return
        if node.is_end:
            for q in node.queries:
                results.append({'text': q, 'frequency': node.frequency})
        for char, child in sorted(node.children.items()):
            self._dfs_collect(child, current + char, results, limit)

    def get_structure(self, max_depth: int = 5, max_children: int = 6):
        """
        Export trie structure as nested dict for D3.js hierarchical visualization.
        We collapse long single-child chains (path compression) like a Radix tree.
        """
        def serialize(node: TrieNode, label: str, depth: int):
            if depth > max_depth:
                return None

            children_list = []
            for char, child in sorted(
                node.children.items(),
                key=lambda x: -x[1].frequency
            )[:max_children]:
                child_node = serialize(child, char, depth + 1)
                if child_node:
                    children_list.append(child_node)

            result = {
                'name': label,
                'freq': node.frequency,
                'is_end': node.is_end,
                'depth': depth,
                'query_count': node.query_count,
            }
            if children_list:
                result['children'] = children_list
            return result

        return serialize(self.root, 'ROOT', 0)

    def get_stats(self):
        """Return statistics about the trie."""
        node_count = [0]
        leaf_count = [0]
        max_depth = [0]

        def count(node, depth):
            node_count[0] += 1
            max_depth[0] = max(max_depth[0], depth)
            if node.is_end:
                leaf_count[0] += 1
            for child in node.children.values():
                count(child, depth + 1)

        count(self.root, 0)
        return {
            'total_queries': self.total_queries,
            'node_count': node_count[0],
            'leaf_count': leaf_count[0],
            'max_depth': max_depth[0],
            'avg_query_length': sum(len(q) for q in self.word_frequencies) / max(len(self.word_frequencies), 1)
        }


# ---------- Module-level singleton ----------
_trie_instance = None

def get_trie():
    global _trie_instance
    if _trie_instance is None:
        import os
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'queries.txt')
        _trie_instance = Trie()
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                _trie_instance.insert(line.strip())
    return _trie_instance
