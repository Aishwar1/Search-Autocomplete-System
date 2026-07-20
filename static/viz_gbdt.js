/* ══════════════════════════════════════════════════════════════════════════
   GBDT Visualization — D3.js Decision Tree + Feature Importance Charts
══════════════════════════════════════════════════════════════════════════ */

let _gbdtImportanceChart = null;
let _gbdtBoostChart = null;

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('gbdt-run-btn').addEventListener('click', () => {
    const q = document.getElementById('gbdt-input').value.trim();
    runGBDT(q);
  });
  document.getElementById('gbdt-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') runGBDT(e.target.value.trim());
  });
});

async function runGBDT(query) {
  try {
    const res = await fetch(`/api/gbdt?query=${encodeURIComponent(query)}`);
    const data = await res.json();

    document.getElementById('gbdt-score').textContent =
      data.popularity_score !== undefined ? (data.popularity_score * 100).toFixed(1) + '%' : '—';

    renderGBDTFeatures(data.features || []);
    renderGBDTTree(data.tree);
    renderImportanceChart(data.global_importance || []);
    renderBoostingCurve(data.boosting_curve || []);
  } catch (e) {
    console.error('GBDT error:', e);
  }
}

function renderGBDTFeatures(features) {
  const el = document.getElementById('gbdt-features');
  el.innerHTML = '';
  const maxImp = Math.max(...features.map(f => f.importance), 0.01);

  features.slice(0, 12).forEach(f => {
    const pct = (f.importance / maxImp * 100).toFixed(0);
    const row = document.createElement('div');
    row.className = 'feat-row';
    row.innerHTML = `
      <span class="feat-name" title="${f.feature}">${f.feature}</span>
      <div class="feat-track"><div class="feat-fill" style="width:${pct}%"></div></div>
      <span class="feat-val">${f.importance.toFixed(3)}</span>
    `;
    el.appendChild(row);
  });
}

function renderGBDTTree(tree) {
  if (!tree) return;
  const svg = d3.select('#gbdt-tree-svg');
  svg.selectAll('*').remove();

  const W = svg.node().getBoundingClientRect().width || 400;
  const H = 380;
  svg.attr('height', H);

  const g = svg.append('g').attr('transform', 'translate(20,20)');

  // Convert tree to d3 hierarchy
  const root = d3.hierarchy(tree, d => {
    if (d.type === 'split') return [d.left, d.right].filter(Boolean);
    return null;
  });

  const treeLayout = d3.tree().size([W - 40, H - 60]);
  treeLayout(root);

  // Links
  g.append('g').selectAll('path')
    .data(root.links())
    .join('path')
    .attr('class', 'tree-link')
    .attr('d', d3.linkVertical()
      .x(d => d.x)
      .y(d => d.y));

  // Nodes
  const nodeG = g.append('g').selectAll('g')
    .data(root.descendants())
    .join('g')
    .attr('class', d => 'tree-node ' + (d.data.type || 'leaf'))
    .attr('transform', d => `translate(${d.x},${d.y})`);

  // Rectangles
  nodeG.append('rect')
    .attr('x', -38).attr('y', -18)
    .attr('width', 76).attr('height', 36);

  // Text: feature and threshold for splits; value for leaves
  nodeG.each(function(d) {
    const el = d3.select(this);
    if (d.data.type === 'split') {
      el.append('text')
        .attr('y', -5).attr('text-anchor', 'middle')
        .attr('font-size', 8).attr('fill', '#4f8ef7')
        .text(d.data.feature?.slice(0, 12));
      el.append('text')
        .attr('y', 7).attr('text-anchor', 'middle')
        .attr('font-size', 8).attr('fill', '#94a3b8')
        .text(`≤ ${d.data.threshold}`);
      el.append('text')
        .attr('y', 16).attr('text-anchor', 'middle')
        .attr('font-size', 7).attr('fill', '#475569')
        .text(`n=${d.data.samples}`);
    } else {
      const val = d.data.value || 0;
      el.append('text')
        .attr('y', -4).attr('text-anchor', 'middle')
        .attr('font-size', 9).attr('fill', '#34d399')
        .text('Leaf');
      el.append('text')
        .attr('y', 8).attr('text-anchor', 'middle')
        .attr('font-size', 8).attr('fill', '#94a3b8')
        .text(val.toFixed(4));
    }
  });

  // Edge labels (True/False)
  g.append('g').selectAll('text')
    .data(root.links())
    .join('text')
    .attr('x', d => (d.source.x + d.target.x) / 2 + (d.target === d.source.children?.[0] ? -12 : 12))
    .attr('y', d => (d.source.y + d.target.y) / 2)
    .attr('font-size', 8)
    .attr('fill', '#475569')
    .attr('text-anchor', 'middle')
    .text(d => d.target === d.source.children?.[0] ? 'T' : 'F');
}

function renderImportanceChart(globalImportance) {
  const canvas = document.getElementById('gbdt-importance-chart');
  if (_gbdtImportanceChart) { _gbdtImportanceChart.destroy(); _gbdtImportanceChart = null; }

  const top = globalImportance.slice(0, 8);

  _gbdtImportanceChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: top.map(f => f.feature.replace('_', ' ')),
      datasets: [{
        data: top.map(f => f.importance),
        backgroundColor: top.map((_, i) => `hsl(${200 + i * 20}, 70%, 60%)`),
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ' ' + ctx.raw.toFixed(4)
          }
        }
      },
      scales: {
        x: { ticks: { color: '#475569', font: { size: 9 } }, grid: { color: '#1e2d45' } },
        y: { ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { display: false } }
      }
    }
  });
}

function renderBoostingCurve(steps) {
  const canvas = document.getElementById('gbdt-boost-chart');
  if (_gbdtBoostChart) { _gbdtBoostChart.destroy(); _gbdtBoostChart = null; }

  _gbdtBoostChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: steps.map(s => s.step),
      datasets: [{
        label: 'Deviance (log-loss)',
        data: steps.map(s => s.loss),
        borderColor: '#a78bfa',
        backgroundColor: 'rgba(167,139,250,0.12)',
        borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4
      }]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 10 } } }
      },
      scales: {
        x: {
          title: { display: true, text: 'Boosting Step', color: '#475569', font: { size: 9 } },
          ticks: { color: '#475569', font: { size: 9 } }, grid: { color: '#1e2d45' }
        },
        y: { ticks: { color: '#475569', font: { size: 9 } }, grid: { color: '#1e2d45' } }
      }
    }
  });
}
