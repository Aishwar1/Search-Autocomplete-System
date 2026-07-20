/* ══════════════════════════════════════════════════════════════════════════
   Metrics Dashboard — Training Loss, Perplexity, Architecture
══════════════════════════════════════════════════════════════════════════ */

let _metLossChart = null;
let _metPplChart = null;

async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    renderMetricsCards(data);
    renderLossChart(data);
    renderPerplexityChart(data);
  } catch (e) {
    console.error('Metrics error:', e);
  }
}

function renderMetricsCards(data) {
  const lossCurve = data.loss_curve || [];
  const finalLoss = lossCurve.length ? lossCurve[lossCurve.length - 1] : null;

  setText('met-final-loss', finalLoss !== null ? finalLoss.toFixed(4) : '—');
  setText('met-epochs', data.epochs ?? '—');
  setText('met-batch', data.batch_size ?? '—');
  setText('met-lr', data.learning_rate !== undefined ? data.learning_rate.toExponential(1) : '—');
}

function renderLossChart(data) {
  const canvas = document.getElementById('met-loss-chart');
  if (_metLossChart) { _metLossChart.destroy(); _metLossChart = null; }

  const curve = data.loss_curve || [];
  if (!curve.length) return;

  _metLossChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: curve.map((_, i) => i + 1),
      datasets: [{
        label: data.demo_mode ? 'Loss (demo)' : 'Training Loss',
        data: curve,
        borderColor: '#4f8ef7',
        backgroundColor: 'rgba(79,142,247,0.1)',
        borderWidth: 2.5,
        pointRadius: curve.length > 30 ? 0 : 3,
        fill: true,
        tension: 0.35
      }]
    },
    options: chartOptions('Training Step', 'Cross-Entropy Loss')
  });
}

function renderPerplexityChart(data) {
  const canvas = document.getElementById('met-perplexity-chart');
  if (_metPplChart) { _metPplChart.destroy(); _metPplChart = null; }

  const curve = data.loss_curve || [];
  if (!curve.length) return;

  const ppl = data.perplexity || curve.map(l => Math.min(Math.exp(l), 500));

  _metPplChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: curve.map((_, i) => i + 1),
      datasets: [{
        label: data.demo_mode ? 'Perplexity (demo)' : 'Perplexity',
        data: ppl,
        borderColor: '#34d399',
        backgroundColor: 'rgba(52,211,153,0.1)',
        borderWidth: 2.5,
        pointRadius: ppl.length > 30 ? 0 : 3,
        fill: true,
        tension: 0.35
      }]
    },
    options: chartOptions('Training Step', 'Perplexity (exp(loss))')
  });
}

function chartOptions(xLabel, yLabel) {
  return {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#94a3b8', font: { size: 11 } } }
    },
    scales: {
      x: {
        title: { display: true, text: xLabel, color: '#475569', font: { size: 10 } },
        ticks: { color: '#475569', font: { size: 9 } },
        grid: { color: '#1e2d45' }
      },
      y: {
        title: { display: true, text: yLabel, color: '#475569', font: { size: 10 } },
        ticks: { color: '#475569', font: { size: 9 } },
        grid: { color: '#1e2d45' }
      }
    }
  };
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
