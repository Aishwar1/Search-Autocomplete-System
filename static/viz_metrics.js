/* Metrics Dashboard — Training Loss, Perplexity, Architecture */

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

  _metSetText('met-final-loss', finalLoss !== null ? finalLoss.toFixed(4) : '—');
  _metSetText('met-epochs', data.epochs ?? '—');
  _metSetText('met-batch', data.batch_size ?? '—');
  _metSetText('met-lr', data.learning_rate !== undefined ? data.learning_rate.toExponential(1) : '—');
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
        borderColor: '#5a9ae8',
        backgroundColor: 'rgba(90,154,232,0.07)',
        borderWidth: 1.5,
        pointRadius: curve.length > 30 ? 0 : 3,
        fill: true,
        tension: 0.3
      }]
    },
    options: metChartOptions('Training Step', 'Cross-Entropy Loss')
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
        borderColor: '#52d18a',
        backgroundColor: 'rgba(82,209,138,0.07)',
        borderWidth: 1.5,
        pointRadius: ppl.length > 30 ? 0 : 3,
        fill: true,
        tension: 0.3
      }]
    },
    options: metChartOptions('Training Step', 'Perplexity')
  });
}

function metChartOptions(xLabel, yLabel) {
  return {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#777', font: { size: 11 } } }
    },
    scales: {
      x: {
        title: { display: true, text: xLabel, color: '#444', font: { size: 10 } },
        ticks: { color: '#444', font: { size: 9 } },
        grid: { color: '#1a1a1a' }
      },
      y: {
        title: { display: true, text: yLabel, color: '#444', font: { size: 10 } },
        ticks: { color: '#444', font: { size: 9 } },
        grid: { color: '#1a1a1a' }
      }
    }
  };
}

function _metSetText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
