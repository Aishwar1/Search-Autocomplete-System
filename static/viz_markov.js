/* Markov Chain Visualization — D3.js Force Graph + Heatmap */

let _markovN = 2;
let _markovSimulation = null;

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('markov-run-btn').addEventListener('click', () => {
    const q = document.getElementById('markov-input').value.trim();
    runMarkovModel(q, _markovN);
  });

  document.getElementById('markov-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') runMarkovModel(e.target.value.trim(), _markovN);
  });

  document.querySelectorAll('.ngram-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.ngram-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      _markovN = parseInt(btn.dataset.n);
    });
  });
});

async function runMarkovModel(query, n) {
  try {
    const res = await fetch(`/api/markov?query=${encodeURIComponent(query)}&n=${n}`);
    const data = await res.json();
    renderMarkovGraph(data.transition_graph, query);
    renderMarkovPredictions(data.next_word_predictions || []);
    renderMarkovCompletions(data.completions || []);
    renderMarkovHeatmap(data.transition_matrix || {});
    updateMarkovStats(data.stats || {});
  } catch (e) {
    console.error('Markov error:', e);
  }
}

function updateMarkovStats(stats) {
  _markovSetText('markov-vocab', stats.vocabulary_size ?? '—');
  _markovSetText('markov-contexts', stats.unique_contexts ?? '—');
  _markovSetText('markov-tokens', stats.total_tokens ?? '—');
  _markovSetText('markov-corpus', stats.corpus_size ?? '—');
}

function renderMarkovPredictions(preds) {
  const el = document.getElementById('markov-predictions');
  el.innerHTML = '';
  const maxProb = Math.max(...preds.map(p => p.probability), 0.01);

  preds.slice(0, 8).forEach(p => {
    const pct = (p.probability / maxProb * 100).toFixed(0);
    const row = document.createElement('div');
    row.className = 'mpred-row';
    row.innerHTML = `
      <span class="mpred-word">${_markovEsc(p.word)}</span>
      <div class="mpred-bar-track">
        <div class="mpred-bar-fill" style="width:${pct}%"></div>
      </div>
      <span class="mpred-prob">${(p.probability * 100).toFixed(1)}%</span>
    `;
    el.appendChild(row);
  });
}

function renderMarkovCompletions(completions) {
  const ul = document.getElementById('markov-completions');
  ul.innerHTML = '';
  completions.slice(0, 6).forEach(c => {
    const li = document.createElement('li');
    li.textContent = c.text;
    ul.appendChild(li);
  });
}

function renderMarkovHeatmap(matrixData) {
  const container = document.getElementById('markov-heatmap');
  container.innerHTML = '';

  const words = matrixData.words || [];
  const matrix = matrixData.matrix || [];
  if (!words.length || !matrix.length) {
    container.innerHTML = '<span style="color:#555;font-size:12px">Type a multi-word query to see transition matrix</span>';
    return;
  }

  const table = document.createElement('table');
  const headerRow = document.createElement('tr');
  headerRow.appendChild(document.createElement('th'));
  words.forEach(w => {
    const th = document.createElement('th');
    th.textContent = w;
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  matrix.forEach((row, i) => {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.textContent = words[i];
    tr.appendChild(th);

    row.forEach(val => {
      const td = document.createElement('td');
      td.textContent = val.toFixed(2);
      const intensity = Math.min(1, val);
      const lightness = Math.round(15 + intensity * 30);
      td.style.background = `hsl(210, 20%, ${lightness}%)`;
      td.style.color = intensity > 0.5 ? '#e0e0e0' : '#999';
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });

  container.appendChild(table);
}

function renderMarkovGraph(graphData, query) {
  if (!graphData) return;
  const svg = d3.select('#markov-svg');
  svg.selectAll('*').remove();

  const width = svg.node().getBoundingClientRect().width || 600;
  const height = 420;
  svg.attr('height', height);

  const nodes = (graphData.nodes || []).map(d => ({ ...d }));
  const links = (graphData.edges || []).map(d => ({ ...d }));

  if (!nodes.length) return;

  svg.append('defs').append('marker')
    .attr('id', 'marrow')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 22)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', '#444');

  const g = svg.append('g');

  // Node colors by group — muted palette
  const groupColors = ['#5a9ae8', '#52d18a', '#d4a843', '#9b7de8', '#e05555'];

  const maxFreq = d3.max(nodes, n => n.frequency || 1) || 1;
  const maxWeight = d3.max(links, l => l.weight || 0.1) || 0.1;

  const linkSel = g.append('g').selectAll('line')
    .data(links)
    .join('line')
    .attr('class', 'm-link')
    .attr('stroke', '#333')
    .attr('stroke-width', d => Math.max(1, d.weight / maxWeight * 3))
    .attr('marker-end', 'url(#marrow)');

  const linkLabels = g.append('g').selectAll('text')
    .data(links)
    .join('text')
    .attr('font-size', 9)
    .attr('fill', '#444')
    .attr('text-anchor', 'middle')
    .text(d => `${(d.weight * 100).toFixed(0)}%`);

  const queryWords = new Set((graphData.query_words || []).map(w => w.toLowerCase()));

  const nodeSel = g.append('g').selectAll('g')
    .data(nodes)
    .join('g')
    .attr('class', 'm-node')
    .call(d3.drag()
      .on('start', dragStarted)
      .on('drag', dragged)
      .on('end', dragEnded));

  nodeSel.append('circle')
    .attr('r', d => {
      const freq = d.frequency || 1;
      return Math.max(8, Math.min(26, 8 + (freq / maxFreq) * 18));
    })
    .attr('fill', d => groupColors[d.group % groupColors.length])
    .attr('fill-opacity', d => queryWords.has(d.id) ? 0.9 : 0.5)
    .attr('stroke', d => queryWords.has(d.id) ? '#e0e0e0' : 'transparent')
    .attr('stroke-width', 1.5)
    .on('mouseover', function(event, d) {
      showMarkovTooltip(event, d);
      d3.select(this).attr('fill-opacity', 1);
    })
    .on('mouseout', function(event, d) {
      hideTooltip();
      d3.select(this).attr('fill-opacity', queryWords.has(d.id) ? 0.9 : 0.5);
    });

  nodeSel.append('text')
    .attr('dy', '0.35em')
    .attr('text-anchor', 'middle')
    .attr('font-size', d => queryWords.has(d.id) ? 11 : 9)
    .attr('font-weight', d => queryWords.has(d.id) ? '700' : '400')
    .attr('fill', '#e0e0e0')
    .text(d => d.label);

  if (_markovSimulation) _markovSimulation.stop();

  _markovSimulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(90).strength(0.5))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide(32))
    .on('tick', () => {
      linkSel
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      linkLabels
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2);

      nodeSel.attr('transform', d => `translate(${d.x},${d.y})`);
    });

  svg.call(d3.zoom()
    .scaleExtent([0.3, 3])
    .on('zoom', e => g.attr('transform', e.transform)));

  function dragStarted(event, d) {
    if (!event.active) _markovSimulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
  }
  function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
  function dragEnded(event, d) {
    if (!event.active) _markovSimulation.alphaTarget(0);
    d.fx = null; d.fy = null;
  }
}

function showMarkovTooltip(event, d) {
  const tip = getOrCreateTooltip();
  tip.style.display = 'block';
  tip.style.left = (event.pageX + 12) + 'px';
  tip.style.top  = (event.pageY - 20) + 'px';
  tip.innerHTML = `
    <div style="font-weight:700">"${d.label}"</div>
    <div>Frequency: ${d.frequency}</div>
    <div>Group: ${d.group}</div>
  `;
}

function _markovSetText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function _markovEsc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
