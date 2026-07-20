/* Markov Chain Visualization — Prediction Bars + Completions */

let _markovN = 2;

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
    renderMarkovPredictions(data.next_word_predictions || []);
    renderMarkovCompletions(data.completions || []);
    renderMarkovLiveExplain(query, data.next_word_predictions || [], n);
  } catch (e) {
    console.error('Markov error:', e);
  }
}

function renderMarkovLiveExplain(query, preds, n) {
  const el = document.getElementById('markov-live-explain');
  if (!el) return;

  const words = query.trim().split(/\s+/);
  const contextSize = n - 1;
  const context = words.slice(-contextSize).join(' ') || query.trim();
  const top3 = preds.slice(0, 3);

  if (!top3.length) { el.style.display = 'none'; return; }

  el.style.display = '';
  el.innerHTML = '';

  const intro = document.createElement('span');
  intro.innerHTML = '<strong>After "</strong>';
  el.appendChild(intro);

  const ctxEm = document.createElement('em');
  ctxEm.textContent = context;
  el.appendChild(ctxEm);

  const mid = document.createElement('span');
  mid.innerHTML = `<strong>"</strong> the ${n === 2 ? 'bigram' : 'trigram'} model predicts: `;
  el.appendChild(mid);

  top3.forEach((p, i) => {
    const span = document.createElement('span');
    span.className = 'explain-highlight';
    span.textContent = p.word;
    el.appendChild(span);

    el.appendChild(document.createTextNode(` (${(p.probability * 100).toFixed(1)}%)`));
    if (i < top3.length - 1) el.appendChild(document.createTextNode(', '));
  });
}

function renderMarkovPredictions(preds) {
  const el = document.getElementById('markov-predictions');
  if (!el) return;
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
  if (!ul) return;
  ul.innerHTML = '';
  completions.slice(0, 6).forEach(c => {
    const li = document.createElement('li');
    li.textContent = c.text;
    ul.appendChild(li);
  });
}

function _markovEsc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
