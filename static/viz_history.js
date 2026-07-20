/* ══════════════════════════════════════════════════════════════════════════
   User History Visualization — Timeline, WordCloud, Intent Distribution
══════════════════════════════════════════════════════════════════════════ */

let _histIntentChart = null;
let _histLengthChart = null;
let _histTimingChart = null;

async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const data = await res.json();
    renderHistoryTimeline(data.recent_queries || []);
    renderWordCloud(data.top_words || []);
    renderIntentChart(data.intent_distribution || {});
    renderLengthChart(data.length_histogram || []);
    renderTimingChart(data.query_timing || []);
    updateHistStats(data);
  } catch (e) {
    console.error('History error:', e);
  }
}

function updateHistStats(data) {
  setText('hist-total', data.total_queries ?? 0);
  setText('hist-unique', data.unique_words ?? 0);
  setText('hist-avg-len', data.avg_query_length ? data.avg_query_length + ' chars' : '—');
  setText('hist-duration', data.session_duration ? formatDuration(data.session_duration) : '0s');
}

function renderHistoryTimeline(queries) {
  const container = document.getElementById('hist-timeline');
  container.innerHTML = '';

  if (!queries.length) {
    container.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:12px">No queries yet — use the Search Engine tab</div>';
    return;
  }

  queries.forEach(q => {
    const div = document.createElement('div');
    div.className = 'hist-item';
    div.innerHTML = `
      <div class="hist-text">${escHtml(q.text)}</div>
      <div class="hist-intent">${q.intent}</div>
      <div class="hist-time">${q.time_ago}</div>
    `;
    container.appendChild(div);
  });
}

function renderWordCloud(words) {
  const el = document.getElementById('hist-wordcloud');
  el.innerHTML = '';

  if (!words.length) {
    el.innerHTML = '<span style="color:var(--text-muted);font-size:11px">No words yet</span>';
    return;
  }

  const maxCount = Math.max(...words.map(w => w.count), 1);
  const colors = ['#4f8ef7', '#34d399', '#fbbf24', '#a78bfa', '#f87171', '#22d3ee', '#f472b6'];

  words.forEach((item, i) => {
    const size = 11 + (item.count / maxCount) * 12;
    const span = document.createElement('span');
    span.className = 'wc-word';
    span.textContent = item.word;
    span.style.fontSize = size + 'px';
    span.style.color = colors[i % colors.length];
    span.style.background = colors[i % colors.length] + '22';
    span.title = `${item.word}: ${item.count}×`;
    el.appendChild(span);
  });
}

function renderIntentChart(intents) {
  const canvas = document.getElementById('hist-intent-chart');
  if (_histIntentChart) { _histIntentChart.destroy(); _histIntentChart = null; }

  const labels = Object.keys(intents);
  const values = Object.values(intents);

  if (!labels.length) return;

  const colors = ['#4f8ef7', '#34d399', '#fbbf24', '#a78bfa', '#f87171', '#22d3ee'];

  _histIntentChart = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors.slice(0, labels.length),
        borderColor: '#0b0f1a',
        borderWidth: 3
      }]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#94a3b8', font: { size: 11 }, boxWidth: 12 }
        }
      }
    }
  });
}

function renderLengthChart(hist) {
  const canvas = document.getElementById('hist-length-chart');
  if (_histLengthChart) { _histLengthChart.destroy(); _histLengthChart = null; }

  _histLengthChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: hist.map(h => h.bin),
      datasets: [{
        label: 'Queries',
        data: hist.map(h => h.count),
        backgroundColor: '#4f8ef766',
        borderColor: '#4f8ef7',
        borderWidth: 2,
        borderRadius: 4
      }]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#475569', font: { size: 10 } }, grid: { display: false } },
        y: { ticks: { color: '#475569', font: { size: 10 } }, grid: { color: '#1e2d45' } }
      }
    }
  });
}

function renderTimingChart(timing) {
  const canvas = document.getElementById('hist-timing-chart');
  if (_histTimingChart) { _histTimingChart.destroy(); _histTimingChart = null; }

  if (!timing.length) return;

  _histTimingChart = new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Query',
        data: timing.map((t, i) => ({ x: t.elapsed, y: i + 1 })),
        backgroundColor: '#22d3ee99',
        borderColor: '#22d3ee',
        pointRadius: 5,
        pointHoverRadius: 7
      }]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const t = timing[ctx.dataIndex];
              return t ? t.text : '';
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Elapsed (s)', color: '#475569', font: { size: 10 } },
          ticks: { color: '#475569', font: { size: 9 } },
          grid: { color: '#1e2d45' }
        },
        y: {
          title: { display: true, text: 'Query #', color: '#475569', font: { size: 10 } },
          ticks: { color: '#475569', font: { size: 9 } },
          grid: { color: '#1e2d45' }
        }
      }
    }
  });
}

function formatDuration(seconds) {
  if (seconds < 60) return Math.round(seconds) + 's';
  if (seconds < 3600) return Math.round(seconds / 60) + 'm';
  return (seconds / 3600).toFixed(1) + 'h';
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Refresh every 10s while tab is open
setInterval(() => {
  const activeTab = document.querySelector('.tab-btn.active');
  if (activeTab && activeTab.dataset.tab === 'history') {
    loadHistory();
  }
}, 10000);
