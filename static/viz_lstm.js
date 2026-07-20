/* LSTM Visualization — SVG Cell Architecture + Timeline Chart */

let _lstmData = null;
let _lstmStep = 0;
let _lstmChart = null;

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('lstm-run-btn').addEventListener('click', () => {
    const q = document.getElementById('lstm-input').value.trim();
    runLSTM(q);
  });

  document.getElementById('lstm-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') runLSTM(e.target.value.trim());
  });

  document.getElementById('lstm-prev').addEventListener('click', () => {
    if (!_lstmData) return;
    _lstmStep = Math.max(0, _lstmStep - 1);
    updateLSTMStep();
  });

  document.getElementById('lstm-next').addEventListener('click', () => {
    if (!_lstmData) return;
    _lstmStep = Math.min((_lstmData.steps.length - 1), _lstmStep + 1);
    updateLSTMStep();
  });
});

function initLSTMViz() {
  drawLSTMArchitecture(null);
}

async function runLSTM(query) {
  const loading = document.getElementById('lstm-loading');
  if (loading) loading.classList.remove('hidden');

  try {
    const res = await fetch(`/api/lstm?query=${encodeURIComponent(query)}`);
    _lstmData = await res.json();
    _lstmStep = 0;
    if (loading) loading.classList.add('hidden');
    updateLSTMStep();
    renderLSTMTimeline(_lstmData.steps || []);
    renderLSTMWordSummary(_lstmData);
  } catch (e) {
    if (loading) { loading.textContent = 'Error: ' + e.message; }
    console.error('LSTM error:', e);
  }
}

function renderLSTMWordSummary(data) {
  const el = document.getElementById('lstm-explain');
  if (!el) return;

  const steps = data.steps || [];
  if (!steps.length) return;

  const query = data.query || '';
  const maxForget = Math.max(...steps.map(s => s.forget_mean));
  const minForget = Math.min(...steps.map(s => s.forget_mean));
  const mostForgetStep = steps.find(s => s.forget_mean === minForget);
  const mostRetainStep = steps.find(s => s.forget_mean === maxForget);

  const safeCharLabel = c => (c === ' ' ? '·(space)' : `'${c}'`);

  // Build DOM safely — no user text in innerHTML
  el.innerHTML = '';

  const b = document.createElement('strong');
  b.textContent = 'Processing: ';
  el.appendChild(b);

  const qSpan = document.createElement('span');
  qSpan.className = 'explain-highlight';
  qSpan.textContent = query;          // safe — textContent
  el.appendChild(qSpan);

  const info = document.createElement('span');
  // Only static/numeric values go into innerHTML here
  info.innerHTML =
    ` — <strong>${steps.length}</strong> character steps computed using real LSTM weights (sigmoid + tanh). ` +
    `The forget gate was <strong>lowest</strong> at ` +
    `<span class="explain-highlight">${safeCharLabel(mostForgetStep?.char || '?')}</span> ` +
    `(model reset memory) and <strong>highest</strong> at ` +
    `<span class="explain-highlight">${safeCharLabel(mostRetainStep?.char || '?')}</span> ` +
    `(model retained context). Use &lt; &gt; to step through each character.`;
  el.appendChild(info);
}

function updateLSTMStep() {
  if (!_lstmData || !_lstmData.steps.length) return;
  const step = _lstmData.steps[_lstmStep];
  if (!step) return;

  const charLabel = step.char === ' ' ? '·' : step.char;
  document.getElementById('lstm-step-label').textContent =
    `t=${_lstmStep} '${charLabel}' (${_lstmStep + 1}/${_lstmData.steps.length})`;

  setBar('g-forget', step.forget_mean, step.forget_mean.toFixed(3));
  setBar('g-input',  step.input_mean,  step.input_mean.toFixed(3));
  setBar('g-cell',   (step.gate_mean + 1) / 2, step.gate_mean.toFixed(3));
  setBar('g-output', step.output_mean, step.output_mean.toFixed(3));

  const maxCellNorm = Math.max(..._lstmData.steps.map(s => s.cell_norm), 1);
  setBar('g-cell-norm', step.cell_norm / maxCellNorm, step.cell_norm.toFixed(3));

  drawLSTMArchitecture(step);
}

function setBar(fillId, pct, label) {
  const fillEl = document.getElementById(fillId);
  const valEl  = document.getElementById(fillId + '-val');
  if (fillEl) fillEl.style.width = (Math.max(0, Math.min(1, pct)) * 100).toFixed(1) + '%';
  if (valEl && label !== undefined) valEl.textContent = label;
}

// ── SVG Cell Diagram ──────────────────────────────────────────────────────────
function drawLSTMArchitecture(step) {
  const svg = d3.select('#lstm-svg');
  svg.selectAll('*').remove();

  const W = svg.node().getBoundingClientRect().width || 560;
  const H = 340;
  svg.attr('height', H);

  const g = svg.append('g');

  // Colors for each gate
  const gateColors = {
    forget: step ? lerpColor('#330000', '#e05555', step.forget_mean) : '#e05555',
    input:  step ? lerpColor('#003333', '#4ecdc4', step.input_mean)  : '#4ecdc4',
    gate:   step ? lerpColor('#332200', '#d4a843', (step.gate_mean+1)/2) : '#d4a843',
    output: step ? lerpColor('#003300', '#52d18a', step.output_mean) : '#52d18a',
    cell:   '#9b7de8',
    hidden: '#5a9ae8'
  };

  // Background
  g.append('rect').attr('width', W).attr('height', H).attr('fill', '#0c0c0c');

  const cx = W / 2;

  // Cell state line (horizontal, top)
  g.append('line')
    .attr('x1', 30).attr('y1', 60)
    .attr('x2', W - 30).attr('y2', 60)
    .attr('stroke', gateColors.cell).attr('stroke-width', 2)
    .attr('stroke-dasharray', '5,3');

  g.append('text').attr('x', cx).attr('y', 46)
    .attr('fill', gateColors.cell).attr('font-size', 10).attr('text-anchor', 'middle')
    .text(`Cell State ct${step ? `  ||norm|| = ${step.cell_norm.toFixed(2)}` : ''}`);

  // Gate boxes
  const gates = [
    { label: 'Forget', sublabel: 'sigma', x: cx - 180, y: H/2 - 10, color: gateColors.forget, val: step?.forget_mean },
    { label: 'Input',  sublabel: 'sigma', x: cx - 60,  y: H/2 - 10, color: gateColors.input,  val: step?.input_mean },
    { label: 'Cell',   sublabel: 'tanh',  x: cx + 60,  y: H/2 - 10, color: gateColors.gate,   val: step?.gate_mean },
    { label: 'Output', sublabel: 'sigma', x: cx + 180, y: H/2 - 10, color: gateColors.output, val: step?.output_mean }
  ];

  gates.forEach(gate => {
    const gEl = g.append('g').attr('transform', `translate(${gate.x},${gate.y})`);

    // Gate box — no rounded corners
    gEl.append('rect')
      .attr('x', -30).attr('y', -30).attr('width', 60).attr('height', 60)
      .attr('fill', '#111')
      .attr('stroke', gate.color).attr('stroke-width', 1.5);

    // Activation label
    gEl.append('text')
      .attr('y', 4).attr('text-anchor', 'middle')
      .attr('fill', gate.color).attr('font-size', 11).attr('font-weight', '600')
      .text(gate.sublabel);

    // Value below box
    if (gate.val !== undefined) {
      gEl.append('text')
        .attr('y', 46).attr('text-anchor', 'middle')
        .attr('fill', gate.color).attr('font-size', 10).attr('font-family', 'Consolas, monospace')
        .text(gate.val.toFixed(3));
    }

    // Gate name above box
    gEl.append('text')
      .attr('y', -36).attr('text-anchor', 'middle')
      .attr('fill', '#777').attr('font-size', 9)
      .text(gate.label);

    // Vertical line to cell state
    gEl.append('line')
      .attr('x1', 0).attr('y1', -30)
      .attr('x2', 0).attr('y2', -60)
      .attr('stroke', gate.color + '88').attr('stroke-width', 1.5);

    // Operator circle on cell line
    g.append('circle')
      .attr('cx', gate.x).attr('cy', 60).attr('r', 9)
      .attr('fill', '#111').attr('stroke', gate.color).attr('stroke-width', 1.5);
    g.append('text')
      .attr('x', gate.x).attr('y', 64)
      .attr('fill', gate.color).attr('font-size', 13).attr('text-anchor', 'middle')
      .text(gate.x < cx ? '×' : (gate.x === cx + 60 ? '×' : '+'));
  });

  // Hidden state line (bottom)
  const hiddenY = H - 50;
  g.append('line')
    .attr('x1', 30).attr('y1', hiddenY)
    .attr('x2', W - 30).attr('y2', hiddenY)
    .attr('stroke', gateColors.hidden).attr('stroke-width', 2);

  g.append('text').attr('x', cx).attr('y', hiddenY + 16)
    .attr('fill', gateColors.hidden).attr('font-size', 10).attr('text-anchor', 'middle')
    .text(`Hidden State ht${step ? `  ||norm|| = ${step.hidden_norm.toFixed(2)}` : ''}`);

  // Input label
  g.append('text').attr('x', 20).attr('y', H/2 + 4)
    .attr('fill', '#555').attr('font-size', 10)
    .text(step ? `xt '${step.char === ' ' ? '·' : step.char}'` : 'xt');

  // Step info
  if (step) {
    g.append('text').attr('x', W - 8).attr('y', 18)
      .attr('fill', '#555').attr('font-size', 10).attr('text-anchor', 'end')
      .text(`step ${_lstmStep + 1} / ${_lstmData?.steps.length || '?'}`);
  }
}

// ── Timeline Chart ────────────────────────────────────────────────────────────
function renderLSTMTimeline(steps) {
  const canvas = document.getElementById('lstm-timeline');
  if (_lstmChart) { _lstmChart.destroy(); _lstmChart = null; }

  const labels = steps.map(s => s.char === ' ' ? '·' : s.char);

  _lstmChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Forget ft',
          data: steps.map(s => s.forget_mean),
          borderColor: '#e05555', backgroundColor: 'transparent',
          borderWidth: 1.5, pointRadius: 2, tension: 0.3
        },
        {
          label: 'Input it',
          data: steps.map(s => s.input_mean),
          borderColor: '#4ecdc4', backgroundColor: 'transparent',
          borderWidth: 1.5, pointRadius: 2, tension: 0.3
        },
        {
          label: 'Output ot',
          data: steps.map(s => s.output_mean),
          borderColor: '#52d18a', backgroundColor: 'transparent',
          borderWidth: 1.5, pointRadius: 2, tension: 0.3
        },
        {
          label: 'Cell Norm',
          data: steps.map(s => s.cell_norm / Math.max(...steps.map(x => x.cell_norm), 1)),
          borderColor: '#9b7de8', backgroundColor: 'rgba(155,125,232,0.06)',
          borderWidth: 1.5, pointRadius: 1, tension: 0.3, fill: true
        }
      ]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#777', font: { size: 10 } } }
      },
      scales: {
        x: { ticks: { color: '#444', font: { size: 9 } }, grid: { color: '#1a1a1a' } },
        y: {
          ticks: { color: '#444', font: { size: 9 } },
          grid: { color: '#1a1a1a' },
          min: 0, max: 1
        }
      },
      onClick: (e, elements) => {
        if (elements.length) {
          _lstmStep = elements[0].index;
          updateLSTMStep();
        }
      }
    }
  });
}

function lerpColor(a, b, t) {
  t = Math.max(0, Math.min(1, t));
  const c = (hex) => {
    const v = parseInt(hex.slice(1), 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  };
  const ca = c(a), cb = c(b);
  const r  = Math.round(ca[0] + (cb[0] - ca[0]) * t);
  const g2 = Math.round(ca[1] + (cb[1] - ca[1]) * t);
  const bl = Math.round(ca[2] + (cb[2] - ca[2]) * t);
  return `#${r.toString(16).padStart(2,'0')}${g2.toString(16).padStart(2,'0')}${bl.toString(16).padStart(2,'0')}`;
}
