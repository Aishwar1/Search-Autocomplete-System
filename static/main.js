/* QueryMind Research Lab — Main Controller */

// ── Search Engine ────────────────────────────────────────────────────────────
const mainSearch    = document.getElementById('main-search');
const dropdown      = document.getElementById('search-dropdown');
const spinner       = document.getElementById('search-spinner');
const trieResults   = document.getElementById('trie-results');
const markovResults = document.getElementById('markov-results');
const tfResults     = document.getElementById('transformer-results');
const tokenDisplay  = document.getElementById('token-display');

let searchDebounce;

mainSearch.addEventListener('input', () => {
  clearTimeout(searchDebounce);
  const q = mainSearch.value.trim();

  if (q.length < 2) {
    dropdown.classList.add('hidden');
    clearResults();
    return;
  }

  spinner.classList.remove('hidden');
  searchDebounce = setTimeout(() => doSearch(q), 280);
});

mainSearch.addEventListener('keydown', e => {
  if (e.key === 'Escape') dropdown.classList.add('hidden');
});

document.addEventListener('click', e => {
  if (!e.target.closest('.search-wrapper')) dropdown.classList.add('hidden');
});

async function doSearch(query) {
  const t0 = performance.now();
  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const data = await res.json();
    const elapsed = performance.now() - t0;

    spinner.classList.add('hidden');
    renderSearchResults(query, data, elapsed);
    updatePipeline(query, data);
    renderDropdown(data);
  } catch (err) {
    spinner.classList.add('hidden');
    console.error('Search error:', err);
  }
}

function renderSearchResults(query, data, elapsed) {
  renderResultList(trieResults, data.trie || [], 'trie-fill', 'freq');
  document.getElementById('trie-time').textContent = `${elapsed.toFixed(0)}ms`;

  renderResultList(markovResults, data.markov || [], 'markov-fill', 'confidence');
  document.getElementById('markov-time').textContent = 'N-gram';

  renderResultList(tfResults, data.transformer || [], 'transformer-fill', 'confidence');
  document.getElementById('transformer-time').textContent =
    data.is_finetuned ? 'Fine-tuned' : 'Base GPT-2';

  renderTokens(data.tokens || []);
}

function renderResultList(ul, items, fillClass, scoreKey) {
  ul.innerHTML = '';
  if (!items.length) {
    ul.innerHTML = '<li style="padding:10px;color:#555;font-size:12px;">No results — keep typing...</li>';
    return;
  }

  items.forEach(item => {
    const text  = item.text || item.query || '';
    const score = item[scoreKey] || item.confidence || item.frequency || 0;
    const pct   = Math.min(100, score * 100).toFixed(1);

    const li = document.createElement('li');
    li.innerHTML = `
      <div class="res-row">
        <span class="res-text">${escHtml(text)}</span>
        <span class="res-conf">${pct}%</span>
      </div>
      <div class="res-bar-track">
        <div class="res-bar-fill ${fillClass}" style="width:${pct}%"></div>
      </div>
    `;
    li.addEventListener('click', () => {
      mainSearch.value = text;
      dropdown.classList.add('hidden');
      doSearch(text);
    });
    ul.appendChild(li);
  });
}

function renderDropdown(data) {
  dropdown.innerHTML = '';
  const all = [
    ...(data.trie || []).slice(0, 3).map(x => ({ ...x, src: 'Trie' })),
    ...(data.markov || []).slice(0, 2).map(x => ({ ...x, src: 'Markov' })),
    ...(data.transformer || []).slice(0, 3).map(x => ({ ...x, src: 'GPT-2' }))
  ];

  if (!all.length) { dropdown.classList.add('hidden'); return; }

  all.forEach(item => {
    const text = item.text || '';
    const div = document.createElement('div');
    div.className = 'dropdown-item';
    div.innerHTML = `
      <span class="di-text">${escHtml(text)}</span>
      <span class="di-source">${item.src}</span>
    `;
    div.addEventListener('click', () => {
      mainSearch.value = text;
      dropdown.classList.add('hidden');
      doSearch(text);
    });
    dropdown.appendChild(div);
  });

  dropdown.classList.remove('hidden');
}

function renderTokens(tokens) {
  tokenDisplay.innerHTML = '';
  if (!tokens.length) {
    tokenDisplay.innerHTML = '<span class="token-placeholder">Type a query above to see tokenization</span>';
    return;
  }

  tokens.forEach((t, i) => {
    const clean = t.replace('Ġ', '·');
    const chip = document.createElement('span');
    chip.className = 'token-chip' + (i >= tokens.length - 2 ? ' last-2' : '');
    chip.textContent = clean;
    chip.title = `Token ${i}: "${t}"`;
    tokenDisplay.appendChild(chip);
  });
}

function updatePipeline(query, data) {
  document.getElementById('pipe-query-val').textContent = query.slice(0, 30);
  document.getElementById('pipe-token-val').textContent =
    (data.tokens || []).length + ' tokens';
  document.getElementById('pipe-topk-val').textContent =
    (data.transformer || []).length + ' candidates';

  const pipeIds = ['pipe-input','pipe-tokenize','pipe-embed','pipe-attn','pipe-topk','pipe-rank'];
  document.querySelectorAll('.pipe-node').forEach(n => n.classList.remove('active'));
  pipeIds.forEach((id, i) => {
    setTimeout(() => {
      document.querySelectorAll('.pipe-node').forEach(n => n.classList.remove('active'));
      const el = document.getElementById(id);
      if (el) el.classList.add('active');
    }, i * 200);
  });
}

function clearResults() {
  [trieResults, markovResults, tfResults].forEach(ul => { ul.innerHTML = ''; });
  tokenDisplay.innerHTML = '<span class="token-placeholder">Type a query above to see tokenization</span>';
}

// ── Header stats ─────────────────────────────────────────────────────────────
async function loadHeaderStats() {
  try {
    const res = await fetch('/api/markov?query=how+to+learn&n=2');
    const data = await res.json();
    const stats = data.stats || {};
    document.querySelector('#stat-corpus .stat-val').textContent =
      (stats.corpus_size || '—').toString();
    document.querySelector('#stat-vocab .stat-val').textContent =
      (stats.vocabulary_size || '—').toString();
    document.querySelector('#stat-model .stat-val').textContent = 'GPT-2';
  } catch (e) {}
}

// ── Utility ──────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Init all sections on load ─────────────────────────────────────────────────
loadHeaderStats();

// Markov — run default query
runMarkovModel('how to learn', 2);

// 3D Embeddings — init viewer and load default words
initEmbeddings3D();
loadEmbeddings();

// Gradient Descent — init viewer and load surface
initGradientViz();
loadGradientSurface();

// Decision Trees — run default query
runGBDT('how to learn python');
