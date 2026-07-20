/* Trie Visualization — D3.js Radial Tree */

let _trieData = null;

function initTrieViz() {
  document.getElementById('trie-search-btn').addEventListener('click', () => {
    const q = document.getElementById('trie-input').value.trim();
    runTrieSearch(q);
  });
  document.getElementById('trie-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') runTrieSearch(e.target.value.trim());
  });
}

async function runTrieSearch(prefix) {
  try {
    const res = await fetch(`/api/trie?prefix=${encodeURIComponent(prefix)}`);
    _trieData = await res.json();
    renderTrie(_trieData, prefix);
    updateTrieStats(_trieData);
    renderTrieLiveExplain(prefix, _trieData);
  } catch (e) {
    console.error('Trie error:', e);
  }
}

function renderTrieLiveExplain(prefix, data) {
  const el = document.getElementById('trie-explain');
  if (!el) return;

  if (!prefix) {
    el.innerHTML = '<strong>How it works:</strong> Type a prefix and click Search. The trie walks down character by character and collects all matching queries at the leaf nodes.';
    return;
  }

  const path = (data.traversal_path || []).map(s => s.char === ' ' ? '[sp]' : s.char);
  const suggestions = data.suggestions || [];
  const count = suggestions.length;
  const nodesVisited = data.nodes_visited ?? '—';

  // Build path spans safely via DOM
  el.innerHTML = '';

  const intro = document.createElement('span');
  intro.innerHTML = '<strong>You searched for "</strong>';
  el.appendChild(intro);

  const em = document.createElement('em');
  em.textContent = prefix;          // safe — textContent never executes HTML
  el.appendChild(em);

  const rest = document.createElement('span');
  rest.innerHTML = '<strong>"</strong> — the trie traversed: ';
  el.appendChild(rest);

  if (path.length) {
    path.forEach((c, i) => {
      const span = document.createElement('span');
      span.className = 'explain-highlight';
      span.textContent = c;          // safe
      el.appendChild(span);
      if (i < path.length - 1) {
        el.appendChild(document.createTextNode(' → '));
      }
    });
  } else {
    el.appendChild(document.createTextNode('(root)'));
  }

  const summary = document.createElement('span');
  summary.innerHTML =
    ` and found <strong>${count} matching quer${count === 1 ? 'y' : 'ies'}</strong> in the subtree. ` +
    `Nodes visited: <strong>${Number(nodesVisited) || nodesVisited}</strong> (green nodes = end of a full query).`;
  el.appendChild(summary);
}

function updateTrieStats(data) {
  const stats = data.stats || {};
  setTrieText('trie-nodes-visited', data.nodes_visited ?? '—');
  setTrieText('trie-subtree', data.subtree_size ?? '—');
  setTrieText('trie-total-nodes', stats.node_count ?? '—');
  setTrieText('trie-max-depth', stats.max_depth ?? '—');
  setTrieText('trie-total-queries', stats.total_queries ?? '—');

  const pathEl = document.getElementById('trie-path-chars');
  pathEl.innerHTML = '';
  (data.traversal_path || []).forEach(step => {
    const span = document.createElement('span');
    span.className = 'traverse-char';
    span.textContent = step.char === ' ' ? '[sp]' : step.char;
    span.title = `freq: ${step.freq}, depth: ${step.depth}`;
    pathEl.appendChild(span);
  });

  const ul = document.getElementById('trie-suggestions');
  ul.innerHTML = '';
  (data.suggestions || []).slice(0, 8).forEach(s => {
    const li = document.createElement('li');
    li.textContent = s.text;
    ul.appendChild(li);
  });
}

function renderTrie(data, prefix) {
  const structure = data.structure;
  if (!structure) return;

  const svg = d3.select('#trie-svg');
  svg.selectAll('*').remove();

  const width = svg.node().getBoundingClientRect().width || 600;
  const height = 520;
  const cx = width / 2;
  const cy = height / 2;

  svg.attr('height', height);

  const g = svg.append('g').attr('transform', `translate(${cx},${cy})`);

  const root = d3.hierarchy(structure, d => d.children);
  const radius = Math.min(width, height) / 2 - 50;

  const tree = d3.tree()
    .size([2 * Math.PI, radius])
    .separation((a, b) => (a.parent === b.parent ? 1 : 2) / a.depth);

  tree(root);

  const traversalChars = new Set((data.traversal_path || []).map(s => s.char));

  g.append('g').selectAll('path')
    .data(root.links())
    .join('path')
    .attr('class', d => {
      const isTraversed = traversalChars.has(d.target.data.name);
      return 'trie-link' + (isTraversed ? ' traversed' : '');
    })
    .attr('d', d3.linkRadial()
      .angle(d => d.x)
      .radius(d => d.y));

  const node = g.append('g').selectAll('g')
    .data(root.descendants())
    .join('g')
    .attr('class', d => {
      let cls = 'trie-node';
      if (d.data.is_end) cls += ' end';
      if (traversalChars.has(d.data.name)) cls += ' traversed';
      return cls;
    })
    .attr('transform', d => `rotate(${d.x * 180 / Math.PI - 90}) translate(${d.y},0)`);

  const maxFreq = d3.max(root.descendants(), d => d.data.freq || 1) || 1;

  node.append('circle')
    .attr('r', d => {
      if (d.depth === 0) return 10;
      const freq = d.data.freq || 1;
      return Math.max(4, Math.min(9, 4 + (freq / maxFreq) * 5));
    })
    .on('mouseover', function(event, d) {
      d3.select(this).attr('stroke-width', 2.5);
      showTrieTooltip(event, d);
    })
    .on('mouseout', function() {
      d3.select(this).attr('stroke-width', 1.5);
      hideTooltip();
    });

  node.append('text')
    .attr('dy', '0.31em')
    .attr('x', d => d.x < Math.PI === !d.children ? 7 : -7)
    .attr('text-anchor', d => d.x < Math.PI === !d.children ? 'start' : 'end')
    .attr('transform', d => d.x >= Math.PI ? 'rotate(180)' : null)
    .text(d => {
      const name = d.data.name;
      return name === 'ROOT' ? 'root' : name === ' ' ? '·' : name;
    })
    .style('font-size', d => d.depth === 0 ? '11px' : '9px')
    .style('fill', d => {
      if (d.data.is_end) return '#52d18a';
      if (traversalChars.has(d.data.name)) return '#5a9ae8';
      return '#555555';
    });
}

function showTrieTooltip(event, d) {
  const tip = getOrCreateTooltip();
  tip.style.display = 'block';
  tip.style.left = (event.pageX + 12) + 'px';
  tip.style.top  = (event.pageY - 20) + 'px';
  tip.innerHTML = `
    <div style="font-weight:600">"${d.data.name === 'ROOT' ? '(root)' : d.data.name}"</div>
    <div>Frequency: ${d.data.freq || 0}</div>
    <div>End of word: ${d.data.is_end ? 'yes' : 'no'}</div>
    <div>Depth: ${d.depth}</div>
  `;
}

function setTrieText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

let _tooltip = null;
function getOrCreateTooltip() {
  if (!_tooltip) {
    _tooltip = document.createElement('div');
    _tooltip.style.cssText = `
      position:fixed;background:#111;border:1px solid #333;
      padding:8px 12px;font-size:12px;color:#e0e0e0;
      pointer-events:none;z-index:9999;font-family:Consolas,monospace;
    `;
    document.body.appendChild(_tooltip);
  }
  return _tooltip;
}

function hideTooltip() {
  if (_tooltip) _tooltip.style.display = 'none';
}
