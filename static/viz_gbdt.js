/* GBDT Visualization — Feature Importance + Score */

let _gbdtImportanceChart = null;

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

    const score = data.popularity_score;
    document.getElementById('gbdt-score').textContent =
      score !== undefined ? (score * 100).toFixed(1) + '%' : '—';

    renderGBDTFeatures(data.features || []);
    renderImportanceChart(data.global_importance || []);
    renderGBDTLiveExplain(query, data);
  } catch (e) {
    console.error('GBDT error:', e);
  }
}

function renderGBDTLiveExplain(query, data) {
  const el = document.getElementById('gbdt-live-explain');
  if (!el) return;

  const features = data.features || [];
  const score = data.popularity_score;
  const top2 = features.slice(0, 2);

  if (!top2.length) { el.style.display = 'none'; return; }

  el.style.display = '';
  el.innerHTML = '';

  const scorePct = (score * 100).toFixed(1);
  const verdictEl = document.createElement('strong');
  if (score >= 0.5) {
    verdictEl.style.color = 'var(--green)';
    verdictEl.textContent = 'Popular query';
  } else {
    verdictEl.style.color = 'var(--amber)';
    verdictEl.textContent = 'Less popular query';
  }
  el.appendChild(verdictEl);
  el.appendChild(document.createTextNode(` (${scorePct}% score) — top features: `));

  top2.forEach((f, i) => {
    const span = document.createElement('span');
    span.className = 'explain-highlight';
    span.textContent = f.feature.replace(/_/g, ' ');
    el.appendChild(span);
    el.appendChild(document.createTextNode(` (${(f.importance * 100).toFixed(1)}%)`));
    if (i < top2.length - 1) el.appendChild(document.createTextNode(' and '));
  });
}

function renderGBDTFeatures(features) {
  const el = document.getElementById('gbdt-features');
  if (!el) return;
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

function renderImportanceChart(globalImportance) {
  const canvas = document.getElementById('gbdt-importance-chart');
  if (!canvas) return;
  if (_gbdtImportanceChart) { _gbdtImportanceChart.destroy(); _gbdtImportanceChart = null; }

  const top = globalImportance.slice(0, 8);

  _gbdtImportanceChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: top.map(f => f.feature.replace(/_/g, ' ')),
      datasets: [{
        data: top.map(f => f.importance),
        backgroundColor: top.map(() => '#5a9ae866'),
        borderColor: top.map(() => '#5a9ae8'),
        borderWidth: 1
      }]
    },
    options: {
      indexAxis: 'y',
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ' ' + ctx.raw.toFixed(4) } }
      },
      scales: {
        x: { ticks: { color: '#444', font: { size: 9 } }, grid: { color: '#1a1a1a' } },
        y: { ticks: { color: '#888', font: { size: 9 } }, grid: { display: false } }
      }
    }
  });
}
