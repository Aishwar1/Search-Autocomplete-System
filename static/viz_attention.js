/* ══════════════════════════════════════════════════════════════════════════
   Attention Map Visualization — Canvas Heatmap (Viridis colormap)
══════════════════════════════════════════════════════════════════════════ */

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
    loading.classList.add('hidden');
    loading.classList.remove('hidden');
    loading.textContent = '⚠ ' + e.message;
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
  ctx.fillStyle = '#131929';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw heatmap cells
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = matrix[i] ? matrix[i][j] || 0 : 0;
      ctx.fillStyle = viridisColor(val);
      ctx.fillRect(
        labelPad + j * cellSize,
        labelPad + i * cellSize,
        cellSize - 1,
        cellSize - 1
      );

      // Value text for small grids
      if (n <= 8 && cellSize >= 40) {
        ctx.fillStyle = val > 0.3 ? '#000' : '#fff';
        ctx.font = `${Math.min(10, cellSize * 0.3)}px Consolas`;
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
  ctx.fillStyle = '#94a3b8';
  ctx.font = `${Math.min(11, cellSize * 0.35)}px Segoe UI`;
  ctx.textAlign = 'right';
  tokens.forEach((tok, j) => {
    ctx.save();
    ctx.translate(labelPad + j * cellSize + cellSize / 2, labelPad - 6);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(tok.slice(0, 8), 0, 0);
    ctx.restore();
  });

  // Token labels — Y axis (left)
  ctx.textAlign = 'right';
  tokens.forEach((tok, i) => {
    ctx.fillText(
      tok.slice(0, 10),
      labelPad - 6,
      labelPad + i * cellSize + cellSize / 2 + 4
    );
  });

  // Colorbar
  drawColorbar(ctx, canvas.width - 20, labelPad, 16, n * cellSize);
}

function drawColorbar(ctx, x, y, w, h) {
  for (let i = 0; i < h; i++) {
    const t = 1 - i / h;
    ctx.fillStyle = viridisColor(t);
    ctx.fillRect(x, y + i, w, 1);
  }
  ctx.strokeStyle = '#263347';
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, w, h);

  ctx.fillStyle = '#94a3b8';
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
    const cs = Math.min(30, Math.floor(140 / Math.max(n, 1)));
    c.width = n * cs;
    c.height = n * cs;
    c.style.width = c.width + 'px';
    c.style.height = c.height + 'px';

    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0b0f1a';
    ctx.fillRect(0, 0, c.width, c.height);

    if (matrix) {
      for (let row = 0; row < n; row++) {
        for (let col = 0; col < n; col++) {
          const val = matrix[row]?.[col] || 0;
          ctx.fillStyle = viridisColor(val);
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

function viridisColor(t) {
  t = Math.max(0, Math.min(1, t));
  // Viridis lookup
  const r = [68,72,67,56,45,37,30,43,81,132,186,253];
  const g = [1,40,90,125,155,184,211,229,243,253,222,231];
  const b = [84,115,140,140,130,121,102,88,73,37,30,37];
  const i = Math.min(10, Math.floor(t * 11));
  const f = t * 11 - i;
  const ri = Math.round(r[i] + (r[i+1]-r[i])*f);
  const gi = Math.round(g[i] + (g[i+1]-g[i])*f);
  const bi = Math.round(b[i] + (b[i+1]-b[i])*f);
  return `rgb(${ri},${gi},${bi})`;
}
