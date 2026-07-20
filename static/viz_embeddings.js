/* 3D Word Embeddings — Three.js Interactive Point Cloud */

let _embedScene, _embedCamera, _embedRenderer, _embedAnimId;
let _embedPoints = [];
let _embedData = null;

function initEmbeddings3D() {
  if (typeof THREE === 'undefined') return;

  const container = document.getElementById('embed-3d-container');
  const W = container.clientWidth || 640;
  const H = 480;

  _embedScene = new THREE.Scene();
  _embedScene.background = new THREE.Color(0x0c0c0c);

  _embedCamera = new THREE.PerspectiveCamera(60, W / H, 0.01, 100);
  _embedCamera.position.set(2.5, 1.5, 2.5);
  _embedCamera.lookAt(0, 0, 0);

  _embedRenderer = new THREE.WebGLRenderer({ antialias: false });
  _embedRenderer.setSize(W, H);
  container.appendChild(_embedRenderer.domElement);

  // Grid
  const grid = new THREE.GridHelper(4, 20, 0x222222, 0x1a1a1a);
  _embedScene.add(grid);

  // Axes
  addEmbedAxes(_embedScene);

  _embedScene.add(new THREE.AmbientLight(0xffffff, 0.8));

  // Mouse orbit
  let isDragging = false, prevMouse = { x: 0, y: 0 };
  let spherical = { theta: 0.8, phi: 1.0, r: 4 };
  let autoRotate = true;

  function updateCamera() {
    _embedCamera.position.x = spherical.r * Math.sin(spherical.phi) * Math.sin(spherical.theta);
    _embedCamera.position.y = spherical.r * Math.cos(spherical.phi);
    _embedCamera.position.z = spherical.r * Math.sin(spherical.phi) * Math.cos(spherical.theta);
    _embedCamera.lookAt(0, 0, 0);
  }
  updateCamera();

  _embedRenderer.domElement.addEventListener('mousedown', e => {
    isDragging = true;
    autoRotate = false;
    prevMouse = { x: e.clientX, y: e.clientY };
  });
  window.addEventListener('mouseup', () => { isDragging = false; });
  window.addEventListener('mousemove', e => {
    if (!isDragging) return;
    spherical.theta -= (e.clientX - prevMouse.x) * 0.008;
    spherical.phi   -= (e.clientY - prevMouse.y) * 0.008;
    spherical.phi    = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
    prevMouse = { x: e.clientX, y: e.clientY };
    updateCamera();
  });
  _embedRenderer.domElement.addEventListener('wheel', e => {
    spherical.r = Math.max(1.5, Math.min(10, spherical.r + e.deltaY * 0.01));
    updateCamera();
    e.preventDefault();
  }, { passive: false });

  function animate() {
    _embedAnimId = requestAnimationFrame(animate);
    if (autoRotate) {
      spherical.theta += 0.003;
      updateCamera();
    }
    _embedRenderer.render(_embedScene, _embedCamera);
  }
  animate();

  // Click to select word
  _embedRenderer.domElement.addEventListener('click', (e) => {
    if (!_embedData) return;
    const rect = _embedRenderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, _embedCamera);
    // Only intersect sphere meshes (not lines/sprites)
    const meshes = _embedPoints.filter(o => o instanceof THREE.Mesh);
    const intersects = raycaster.intersectObjects(meshes, false);
    if (intersects.length > 0) {
      const obj = intersects[0].object;
      if (obj.userData && obj.userData.word) {
        showWordInfo(obj.userData);
      }
    }
  });

  window.addEventListener('resize', () => {
    const W2 = container.clientWidth;
    _embedCamera.aspect = W2 / H;
    _embedCamera.updateProjectionMatrix();
    _embedRenderer.setSize(W2, H);
  });

  document.getElementById('embed-load-btn').addEventListener('click', loadEmbeddings);
}

async function loadEmbeddings() {
  const wordsInput = document.getElementById('embed-words-input').value.trim();
  const url = wordsInput
    ? `/api/embeddings?words=${encodeURIComponent(wordsInput)}`
    : '/api/embeddings';

  try {
    const res = await fetch(url);
    _embedData = await res.json();
    renderEmbedPoints(_embedData);
    updateEmbedStats(_embedData);
    renderEmbedLegend(_embedData);
  } catch (e) {
    console.error('Embeddings error:', e);
  }
}

function renderEmbedPoints(data) {
  if (!_embedScene || !data.points) return;

  _embedPoints.forEach(obj => _embedScene.remove(obj));
  _embedPoints = [];

  data.points.forEach(pt => {
    const color = new THREE.Color(pt.color || '#5a9ae8');

    // Sphere
    const geo = new THREE.SphereGeometry(0.05, 8, 8);
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.8, metalness: 0.0 });
    const sphere = new THREE.Mesh(geo, mat);
    sphere.position.set(pt.x * 1.8, pt.y * 1.8, pt.z * 1.8);
    sphere.userData = pt;
    _embedScene.add(sphere);
    _embedPoints.push(sphere);

    // Label sprite
    const sprite = makeTextSprite(pt.word, color);
    sprite.position.set(pt.x * 1.8, pt.y * 1.8 + 0.13, pt.z * 1.8);
    sprite.userData = pt;
    _embedScene.add(sprite);
    _embedPoints.push(sprite);
  });

  // Connection lines between nearest neighbors
  const pts = data.points;
  pts.forEach(pt => {
    (pt.nearest || []).forEach(nearWord => {
      const nearPt = pts.find(p => p.word === nearWord);
      if (!nearPt) return;
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(pt.x * 1.8, pt.y * 1.8, pt.z * 1.8),
        new THREE.Vector3(nearPt.x * 1.8, nearPt.y * 1.8, nearPt.z * 1.8)
      ]);
      const mat = new THREE.LineBasicMaterial({
        color: new THREE.Color(pt.color), transparent: true, opacity: 0.12
      });
      const line = new THREE.Line(geo, mat);
      _embedScene.add(line);
      _embedPoints.push(line);
    });
  });
}

function makeTextSprite(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 256, 64);
  ctx.font = 'bold 26px Segoe UI';
  ctx.fillStyle = '#' + color.getHexString();
  ctx.textAlign = 'center';
  ctx.fillText(text, 128, 42);

  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.7, 0.18, 1);
  return sprite;
}

function addEmbedAxes(scene) {
  const axisLen = 1.8;
  const axes = [
    { dir: [1,0,0], color: '#e05555', label: 'PC1' },
    { dir: [0,1,0], color: '#52d18a', label: 'PC2' },
    { dir: [0,0,1], color: '#5a9ae8', label: 'PC3' }
  ];
  axes.forEach(ax => {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0,0,0),
      new THREE.Vector3(...ax.dir.map(v => v * axisLen))
    ]);
    const mat = new THREE.LineBasicMaterial({ color: new THREE.Color(ax.color), transparent: true, opacity: 0.35 });
    scene.add(new THREE.Line(geo, mat));
  });
}

function updateEmbedStats(data) {
  const ev = data.explained_variance || [];
  _embedSetText('embed-pc1', ev[0] !== undefined ? (ev[0] * 100).toFixed(1) + '%' : '—');
  _embedSetText('embed-pc2', ev[1] !== undefined ? (ev[1] * 100).toFixed(1) + '%' : '—');
  _embedSetText('embed-pc3', ev[2] !== undefined ? (ev[2] * 100).toFixed(1) + '%' : '—');
  _embedSetText('embed-total-var', data.total_variance_explained !== undefined
    ? (data.total_variance_explained * 100).toFixed(1) + '%' : '—');
  _embedSetText('embed-clusters', data.n_clusters ?? '—');
}

function renderEmbedLegend(data) {
  const el = document.getElementById('embed-legend');
  el.innerHTML = '';
  const clusters = {};
  (data.points || []).forEach(pt => {
    if (!clusters[pt.cluster]) clusters[pt.cluster] = { label: pt.cluster_label, color: pt.color };
  });

  Object.values(clusters).forEach(c => {
    const div = document.createElement('div');
    div.className = 'legend-item';
    div.innerHTML = `
      <div class="legend-dot" style="background:${c.color}"></div>
      <span>${c.label || `Cluster`}</span>
    `;
    el.appendChild(div);
  });
}

function showWordInfo(pt) {
  const el = document.getElementById('embed-nearest');
  el.innerHTML = `
    <div class="traverse-label">Selected: <strong style="color:${pt.color}">${pt.word}</strong></div>
    <div style="font-size:11px;margin-top:6px;color:#888;font-family:Consolas,monospace">
      Cluster: ${pt.cluster_label || pt.cluster}<br>
      PCA: (${pt.x?.toFixed(2)}, ${pt.y?.toFixed(2)}, ${pt.z?.toFixed(2)})<br>
      ${pt.nearest ? 'Nearest: ' + pt.nearest.join(', ') : ''}
    </div>
  `;
}

function _embedSetText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
