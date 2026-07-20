/* ══════════════════════════════════════════════════════════════════════════
   LSTM Visualization — SVG Cell Architecture + Timeline Chart
══════════════════════════════════════════════════════════════════════════ */

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
  drawLSTMArchitecture();
}

async function runLSTM(query) {
  try {
    const res = await fetch(`/api/lstm?query=${encodeURIComponent(query)}`);
    _lstmData = await res.json();
    _lstmStep = 0;
    updateLSTMStep();
    renderLSTMTimeline(_lstmData.steps || []);
  } catch (e) {
    console.error('LSTM error:', e);
  }
}

function updateLSTMStep() {
  if (!_lstmData || !_lstmData.steps.length) return;
  const step = _lstmData.steps[_lstmStep];
  if (!step) return;

  document.getElementById('lstm-step-label').innerHTML =
    `t = ${_lstmStep} '<span id="lstm-char-label">${step.char === ' ' ? '·' : step.char}</span>'`;
  document.getElementById('lstm-char-label').textContent = step.char === ' ' ? '·' : step.char;

  // Update gate bars
  setBar('g-forget', step.forget_mean, step.forget_mean.toFixed(3));
  setBar('g-input',  step.input_mean,  step.input_mean.toFixed(3));
  setBar('g-cell',   (step.gate_mean + 1) / 2, step.gate_mean.toFixed(3)); // tanh → [0,1]
  setBar('g-output', step.output_mean, step.output_mean.toFixed(3));

  const maxCellNorm = Math.max(..._lstmData.steps.map(s => s.cell_norm), 1);
  setBar('g-cell-norm', step.cell_norm / maxCellNorm, step.cell_norm.toFixed(3));

  // Redraw architecture with live values
  drawLSTMArchitecture(step);
}

function setBar(fillId, pct, label) {
  const fillEl = document.getElementById(fillId);
  const valEl  = document.getElementById(fillId + '-val');
  if (fillEl) fillEl.style.width = (Math.max(0, Math.min(1, pct)) * 100).toFixed(1) + '%';
  if (valEl && label !== undefined) valEl.textContent = label;
}

// ── SVG Cell Diagram ─────────────────────────────────────────────────────────
function drawLSTMArchitecture(step) {
  const svg = d3.select('#lstm-svg');
  svg.selectAll('*').remove();

  const W = svg.node().getBoundingClientRect().width || 560;
  const H = 340;
  svg.attr('height', H);

  const g = svg.append('g');

  // Color scheme
  const colors = {
    forget: step ? lerpColor('#7f1d1d', '#f87171', step.forget_mean) : '#f87171',
    input:  step ? lerpColor('#164e63', '#22d3ee', step.input_mean) : '#22d3ee',
    gate:   step ? lerpColor('#78350f', '#fbbf24', (step.gate_mean+1)/2) : '#fbbf24',
    output: step ? lerpColor('#064e3b', '#34d399', step.output_mean) : '#34d399',
    cell:   '#a78bfa',
    hidden: '#4f8ef7'
  };

  // Background
  g.append('rect').attr('width', W).attr('height', H)
    .attr('fill', '#0e1726').attr('rx', 8);

  const cx = W / 2;
  const cy = H / 2;

  // Cell state line (horizontal)
  g.append('line')
    .attr('x1', 40).attr('y1', 60)
    .attr('x2', W - 40).attr('y2', 60)
    .attr('stroke', colors.cell).attr('stroke-width', 3)
    .attr('stroke-dasharray', '6,3');

  // Cell state label
  g.append('text').attr('x', cx).attr('y', 48)
    .attr('fill', colors.cell).attr('font-size', 11).attr('text-anchor', 'middle')
    .text(`Cell State cₜ${step ? ` ‖norm‖= ${step.cell_norm.toFixed(2)}` : ''}`);

  // ── Gate boxes ───────────────────────────────────────────────────────────
  const gates = [
    { label: 'Forget\nGate fₜ', sublabel: 'σ', x: cx - 180, y: cy - 10, color: colors.forget, val: step?.forget_mean },
    { label: 'Input\nGate iₜ',  sublabel: 'σ', x: cx - 60,  y: cy - 10, color: colors.input,  val: step?.input_mean },
    { label: 'Cell\nGate g̃ₜ',  sublabel: 'tanh', x: cx + 60,y: cy - 10, color: colors.gate,   val: step?.gate_mean },
    { label: 'Output\nGate oₜ', sublabel: 'σ', x: cx + 180, y: cy - 10, color: colors.output, val: step?.output_mean }
  ];

  gates.forEach(gate => {
    const gEl = g.append('g').attr('transform', `translate(${gate.x},${gate.y})`);

    // Gate box
    gEl.append('rect')
      .attr('x', -32).attr('y', -32).attr('width', 64).attr('height', 64)
      .attr('rx', 8).attr('fill', gate.color + '22')
      .attr('stroke', gate.color).attr('stroke-width', 2);

    // Activation symbol
    gEl.append('text')
      .attr('y', 5).attr('text-anchor', 'middle')
      .attr('fill', gate.color).attr('font-size', 14).attr('font-weight', '700')
      .text(gate.sublabel);

    // Value badge
    if (gate.val !== undefined) {
      gEl.append('rect')
        .attr('x', -22).attr('y', 38).attr('width', 44).attr('height', 16)
        .attr('rx', 4).attr('fill', gate.color + '33');
      gEl.append('text')
        .attr('y', 50).attr('text-anchor', 'middle')
        .attr('fill', gate.color).attr('font-size', 10)
        .text(gate.val.toFixed(3));
    }

    // Label lines
    const lines = gate.label.split('\n');
    lines.forEach((line, i) => {
      gEl.append('text')
        .attr('y', -38 + i * 12).attr('text-anchor', 'middle')
        .attr('fill', '#94a3b8').attr('font-size', 9)
        .text(line);
    });

    // Vertical line to cell state
    gEl.append('line')
      .attr('x1', 0).attr('y1', -32)
      .attr('x2', 0).attr('y2', -70)
      .attr('stroke', gate.color + '88').attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '3,2');

    // × or + operator on cell line
    const opX = gate.x;
    const opY = 60;
    const opGlyph = gate.x < cx ? '×' : (gate.x === cx + 60 ? '×' : '+');
    g.append('circle')
      .attr('cx', opX).attr('cy', opY).attr('r', 10)
      .attr('fill', '#1a2235').attr('stroke', gate.color).attr('stroke-width', 1.5);
    g.append('text')
      .attr('x', opX).attr('y', opY + 4)
      .attr('fill', gate.color).attr('font-size', 12).attr('text-anchor', 'middle')
      .text(opGlyph);
  });

  // Hidden state arrow (bottom)
  const hiddenY = H - 50;
  g.append('line')
    .attr('x1', 40).attr('y1', hiddenY)
    .attr('x2', W - 40).attr('y2', hiddenY)
    .attr('stroke', colors.hidden).attr('stroke-width', 3);

  g.append('text').attr('x', cx).attr('y', hiddenY + 18)
    .attr('fill', colors.hidden).attr('font-size', 11).attr('text-anchor', 'middle')
    .text(`Hidden State hₜ${step ? ` ‖norm‖= ${step.hidden_norm.toFixed(2)}` : ''}`);

  // Input x arrow
  g.append('text').attr('x', 20).attr('y', cy + 5)
    .attr('fill', '#475569').attr('font-size', 11)
    .text(`xₜ${step ? ` '${step.char === ' ' ? '·' : step.char}'` : ''}`);
  g.append('line')
    .attr('x1', 50).attr('y1', cy)
    .attr('x2', cx - 210).attr('y2', cy)
    .attr('stroke', '#475569').attr('stroke-width', 1.5)
    .attr('marker-end', 'url(#arrowGray)');

  // Define gray arrow
  const defs = svg.select('defs').empty() ? svg.append('defs') : svg.select('defs');
  defs.append('marker')
    .attr('id', 'arrowGray').attr('viewBox', '0 -5 10 10')
    .attr('refX', 8).attr('refY', 0)
    .attr('markerWidth', 5).attr('markerHeight', 5)
    .attr('orient', 'auto')
    .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#475569');

  // Step counter
  if (step) {
    g.append('text').attr('x', W - 10).attr('y', 20)
      .attr('fill', '#4f8ef7').attr('font-size', 12).attr('text-anchor', 'end')
      .text(`Step ${_lstmStep + 1} / ${_lstmData?.steps.length || '?'}`);
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
          label: 'Forget Gate fₜ',
          data: steps.map(s => s.forget_mean),
          borderColor: '#f87171', backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 3, tension: 0.3
        },
        {
          label: 'Input Gate iₜ',
          data: steps.map(s => s.input_mean),
          borderColor: '#22d3ee', backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 3, tension: 0.3
        },
        {
          label: 'Output Gate oₜ',
          data: steps.map(s => s.output_mean),
          borderColor: '#34d399', backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 3, tension: 0.3
        },
        {
          label: 'Cell Norm',
          data: steps.map(s => s.cell_norm / Math.max(...steps.map(x => x.cell_norm), 1)),
          borderColor: '#a78bfa', backgroundColor: 'rgba(167,139,250,0.08)',
          borderWidth: 2, pointRadius: 2, tension: 0.3, fill: true
        }
      ]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 10 } } }
      },
      scales: {
        x: { ticks: { color: '#475569', font: { size: 9 } }, grid: { color: '#1e2d45' } },
        y: {
          ticks: { color: '#475569', font: { size: 9 } },
          grid: { color: '#1e2d45' },
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

// ── Color lerp ────────────────────────────────────────────────────────────────
function lerpColor(a, b, t) {
  t = Math.max(0, Math.min(1, t));
  const c = (hex) => {
    const v = parseInt(hex.slice(1), 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  };
  const ca = c(a), cb = c(b);
  const r = Math.round(ca[0] + (cb[0] - ca[0]) * t);
  const g2 = Math.round(ca[1] + (cb[1] - ca[1]) * t);
  const bl = Math.round(ca[2] + (cb[2] - ca[2]) * t);
  return `#${r.toString(16).padStart(2,'0')}${g2.toString(16).padStart(2,'0')}${bl.toString(16).padStart(2,'0')}`;
}
