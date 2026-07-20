/* Attention Map Visualization — Canvas Heatmap */

let _attnData = null;
let _attnHead = 'avg';
let _attnLayer = 5;

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('attn-run-btn').addEventListener('click', () => {
    const q = document.getElementById('attn-input').value.trim();
    runAttention(q, _attnLayer);
  });

  document.getElementById('attn-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') runAttention(e.target.value.trim(), _attnLayer);
  });

  const slider = document.getElementById('attn-layer-slider');
  slider.addEventListener('input', () => {
    _attnLayer = parseInt(slider.value);
    document.getElementById('attn-layer-val').textContent = _attnLayer;
    document.getElementById('attn-layer-label').textContent = _attnLayer;
    if (_attnData) rerenderAttention();
  });

  document.querySelectorAll('.head-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.head-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      _attnHead = btn.dataset.head;
      if (_attnData) rerenderAttention();
    });
  });
});

async function runAttention(query, layer) {
  const loading = document.getElementById('attn-loading');
  loading.textContent = 'Extracting attention weights...';
  loading.classList.remove('hidden');

  try {
    const res = await fetch(`/api/attention?query=${encodeURIComponent(query)}&layer=${layer}`);
    _attnData = await res.json();

    if (_attnData.error) throw new Error(_attnData.error);

    document.getElementById('attn-token-count').textContent = _attnData.seq_len ?? '—';
    loading.classList.add('hidden');
    rerenderAttention();
    renderAllHeadsMini(_attnData, layer);
  } catch (e) {
    loading.textContent = 'Error: ' + e.message;
    loading.classList.remove('hidden');
    console.error('Attention error:', e);
  }
}

function rerenderAttention() {
  if (!_attnData || !_attnData.attention_by_layer) return;

  const layer = _attnLayer;
  const layerData = _attnData.attention_by_layer[layer];
  if (!layerData) return;

  let matrix;
  if (_attnHead === 'avg') {
    matrix = layerData.avg;
  } else {
    const hIdx = parseInt(_attnHead);
    matrix = layerData.heads[hIdx] || layerData.avg;
  }

  drawAttentionHeatmap(
    document.getElementById('attn-canvas'),
    matrix,
    _attnData.tokens || []
  );
}

function drawAttentionHeatmap(canvas, matrix, tokens) {
  const n = tokens.length;
  if (!n || !matrix) return;

  const cellSize = Math.min(50, Math.floor(Math.min(600, 700) / n));
  const labelPad = 80;

  canvas.width  = labelPad + n * cellSize;
  canvas.height = labelPad + n * cellSize;
  canvas.style.width  = canvas.width  + 'px';
  canvas.style.height = canvas.height + 'px';

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#0c0c0c';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw heatmap cells
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = matrix[i] ? matrix[i][j] || 0 : 0;
      ctx.fillStyle = attnColor(val);
      ctx.fillRect(
        labelPad + j * cellSize,
        labelPad + i * cellSize,
        cellSize - 1,
        cellSize - 1
      );

      if (n <= 8 && cellSize >= 40) {
        ctx.fillStyle = val > 0.4 ? '#000' : '#ccc';
        ctx.font = `${Math.min(10, cellSize * 0.28)}px Consolas`;
        ctx.textAlign = 'center';
        ctx.fillText(
          val.toFixed(2),
          labelPad + j * cellSize + cellSize / 2,
          labelPad + i * cellSize + cellSize / 2 + 3
        );
      }
    }
  }

  // Token labels — X axis (top)
  ctx.fillStyle = '#888';
  ctx.font = `${Math.min(11, cellSize * 0.35)}px Segoe UI`;
  ctx.textAlign = 'right';
  tokens.forEach((tok, j) => {
    ctx.save();
    ctx.translate(labelPad + j * cellSize + cellSize / 2, labelPad - 5);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(tok.slice(0, 8), 0, 0);
    ctx.restore();
  });

  // Token labels — Y axis (left)
  ctx.textAlign = 'right';
  tokens.forEach((tok, i) => {
    ctx.fillText(
      tok.slice(0, 10),
      labelPad - 5,
      labelPad + i * cellSize + cellSize / 2 + 4
    );
  });

  // Colorbar
  drawColorbar(ctx, canvas.width - 18, labelPad, 14, n * cellSize);
}

function drawColorbar(ctx, x, y, w, h) {
  for (let i = 0; i < h; i++) {
    const t = 1 - i / h;
    ctx.fillStyle = attnColor(t);
    ctx.fillRect(x, y + i, w, 1);
  }
  ctx.strokeStyle = '#2a2a2a';
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, w, h);

  ctx.fillStyle = '#888';
  ctx.font = '9px Consolas';
  ctx.textAlign = 'left';
  ctx.fillText('1.0', x + w + 2, y + 8);
  ctx.fillText('0.0', x + w + 2, y + h);
}

function renderAllHeadsMini(data, layer) {
  const grid = document.getElementById('attn-heads-grid');
  grid.innerHTML = '';

  const layerData = data.attention_by_layer?.[layer];
  if (!layerData) return;

  const heads = [layerData.avg, ...(layerData.heads || [])];
  const labels = ['Avg', 'Head 0', 'Head 1', 'Head 2', 'Head 3'];

  heads.forEach((matrix, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'attn-head-mini';

    const c = document.createElement('canvas');
    const n = (data.tokens || []).length;
    const cs = Math.min(28, Math.floor(130 / Math.max(n, 1)));
    c.width = n * cs;
    c.height = n * cs;
    c.style.width = c.width + 'px';
    c.style.height = c.height + 'px';

    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0c0c0c';
    ctx.fillRect(0, 0, c.width, c.height);

    if (matrix) {
      for (let row = 0; row < n; row++) {
        for (let col = 0; col < n; col++) {
          const val = matrix[row]?.[col] || 0;
          ctx.fillStyle = attnColor(val);
          ctx.fillRect(col * cs, row * cs, cs, cs);
        }
      }
    }

    const lbl = document.createElement('div');
    lbl.className = 'attn-head-label';
    lbl.textContent = labels[i] || `Head ${i}`;

    wrap.appendChild(c);
    wrap.appendChild(lbl);
    grid.appendChild(wrap);
  });
}

// Grayscale heatmap — brightest white = max attention, dark = low
function attnColor(t) {
  t = Math.max(0, Math.min(1, t));
  // Use a simple blue-to-white ramp that looks clean on black
  const r = Math.round(20 + t * 200);
  const g = Math.round(30 + t * 160);
  const b = Math.round(50 + t * 200);
  return `rgb(${r},${g},${b})`;
}
