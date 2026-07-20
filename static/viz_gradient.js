/* ══════════════════════════════════════════════════════════════════════════
   Gradient Descent — Three.js 3D Loss Surface + Animated Ball
══════════════════════════════════════════════════════════════════════════ */

let _gradScene, _gradCamera, _gradRenderer, _gradAnimId;
let _gradData = null;
let _gradStep = 0;
let _gradPlaying = false;
let _gradBall = null;
let _gradPath = [];
let _gradLossChart = null;

function initGradientViz() {
  if (typeof THREE === 'undefined') return;

  const container = document.getElementById('grad-3d-container');
  const W = container.clientWidth || 640;
  const H = 500;

  _gradScene = new THREE.Scene();
  _gradScene.background = new THREE.Color(0x0b0f1a);

  _gradCamera = new THREE.PerspectiveCamera(55, W / H, 0.01, 200);
  _gradCamera.position.set(5, 6, 8);
  _gradCamera.lookAt(0, 0, 0);

  _gradRenderer = new THREE.WebGLRenderer({ antialias: true });
  _gradRenderer.setSize(W, H);
  _gradRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(_gradRenderer.domElement);

  // Lighting
  _gradScene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5, 10, 5);
  _gradScene.add(dir);

  // Orbit controls (manual)
  let isDrag = false, prevMouse = { x: 0, y: 0 };
  let sph = { theta: 0.7, phi: 0.9, r: 10 };

  function updateCam() {
    _gradCamera.position.x = sph.r * Math.sin(sph.phi) * Math.sin(sph.theta);
    _gradCamera.position.y = sph.r * Math.cos(sph.phi);
    _gradCamera.position.z = sph.r * Math.sin(sph.phi) * Math.cos(sph.theta);
    _gradCamera.lookAt(0, 0, 0);
  }
  updateCam();

  _gradRenderer.domElement.addEventListener('mousedown', e => { isDrag = true; prevMouse = { x: e.clientX, y: e.clientY }; });
  window.addEventListener('mouseup', () => { isDrag = false; });
  window.addEventListener('mousemove', e => {
    if (!isDrag) return;
    sph.theta -= (e.clientX - prevMouse.x) * 0.009;
    sph.phi   -= (e.clientY - prevMouse.y) * 0.009;
    sph.phi = Math.max(0.2, Math.min(Math.PI * 0.45, sph.phi));
    prevMouse = { x: e.clientX, y: e.clientY };
    updateCam();
  });
  _gradRenderer.domElement.addEventListener('wheel', e => {
    sph.r = Math.max(4, Math.min(20, sph.r + e.deltaY * 0.02));
    updateCam();
  });

  function animate() {
    _gradAnimId = requestAnimationFrame(animate);
    _gradRenderer.render(_gradScene, _gradCamera);
  }
  animate();

  // Resize
  window.addEventListener('resize', () => {
    const W2 = container.clientWidth;
    _gradCamera.aspect = W2 / H;
    _gradCamera.updateProjectionMatrix();
    _gradRenderer.setSize(W2, H);
  });

  // Controls
  document.getElementById('grad-play-btn').addEventListener('click', toggleGradPlay);
  document.getElementById('grad-reset-btn').addEventListener('click', resetGrad);
}

async function loadGradientSurface() {
  try {
    const res = await fetch('/api/gradient');
    _gradData = await res.json();
    buildGradSurface(_gradData);
    buildGradPath(_gradData.gradient_path || []);
    buildGradLossChart(_gradData.gradient_path || []);
    updateGradInfo(0);
  } catch (e) {
    console.error('Gradient error:', e);
  }
}

function buildGradSurface(data) {
  if (!_gradScene) return;

  const w1 = data.w1 || [];
  const w2 = data.w2 || [];
  const surface = data.loss_surface || [];
  const N = w1.length;
  const M = w2.length;

  if (!N || !M) return;

  // Build geometry
  const vertices = [];
  const colors   = [];
  const indices  = [];

  const scaleX = 4 / (N - 1);
  const scaleZ = 4 / (M - 1);

  // Find min/max for color mapping
  let lMin = Infinity, lMax = -Infinity;
  surface.forEach(row => row.forEach(v => {
    lMin = Math.min(lMin, v);
    lMax = Math.max(lMax, v);
  }));

  for (let j = 0; j < M; j++) {
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * 4 - 2;
      const z = (j / (M - 1)) * 4 - 2;
      const loss = surface[j] ? (surface[j][i] || 0) : 0;
      const y = Math.min(loss * 0.4, 3);

      vertices.push(x, y, z);

      // Viridis color
      const t = (loss - lMin) / (lMax - lMin + 1e-10);
      const rgb = viridisRGB(t);
      colors.push(rgb[0], rgb[1], rgb[2]);
    }
  }

  for (let j = 0; j < M - 1; j++) {
    for (let i = 0; i < N - 1; i++) {
      const a = j * N + i;
      const b = j * N + i + 1;
      const c = (j + 1) * N + i;
      const d = (j + 1) * N + i + 1;
      indices.push(a, b, c, b, d, c);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geo.setAttribute('color',    new THREE.Float32BufferAttribute(colors, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true, side: THREE.DoubleSide,
    roughness: 0.8, metalness: 0.1
  });

  const mesh = new THREE.Mesh(geo, mat);
  _gradScene.add(mesh);

  // Wireframe overlay
  const wMat = new THREE.MeshBasicMaterial({ wireframe: true, color: 0x1e2d45, opacity: 0.3, transparent: true });
  _gradScene.add(new THREE.Mesh(geo, wMat));

  // Minimum marker
  const minGeo = new THREE.ConeGeometry(0.12, 0.4, 12);
  const minMat = new THREE.MeshStandardMaterial({ color: 0x34d399, emissive: 0x34d399, emissiveIntensity: 0.5 });
  const minMarker = new THREE.Mesh(minGeo, minMat);
  minMarker.position.set(0, 0.25, 0);
  _gradScene.add(minMarker);

  // Coordinate axes
  const axesMat = (c) => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.5 });
  [
    { pts: [[-2,0,0],[2,0,0]], c: 0xf87171 },
    { pts: [[0,0,0],[0,3,0]], c: 0x34d399 },
    { pts: [[0,0,-2],[0,0,2]], c: 0x4f8ef7 }
  ].forEach(ax => {
    const geo2 = new THREE.BufferGeometry().setFromPoints(ax.pts.map(p => new THREE.Vector3(...p)));
    _gradScene.add(new THREE.Line(geo2, axesMat(ax.c)));
  });
}

function buildGradPath(pathData) {
  if (!_gradScene || !pathData.length) return;
  _gradPath = pathData;

  // Ball
  const ballGeo = new THREE.SphereGeometry(0.12, 16, 16);
  const ballMat = new THREE.MeshStandardMaterial({
    color: 0xfbbf24, emissive: 0xfbbf24, emissiveIntensity: 0.6,
    roughness: 0.3, metalness: 0.5
  });
  _gradBall = new THREE.Mesh(ballGeo, ballMat);
  _gradScene.add(_gradBall);

  // Glow ring
  const ringGeo = new THREE.TorusGeometry(0.18, 0.03, 8, 24);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.4 });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  _gradBall.add(ring);

  // Set initial position
  moveBallTo(pathData[0]);
}

function moveBallTo(pt) {
  if (!_gradBall || !pt) return;
  const x = ((pt.w1 + 2) / 4) * 4 - 2;
  const z = ((pt.w2 - (-1)) / 4) * 4 - 2;
  const y = Math.min(pt.loss * 0.4 + 0.15, 3.15);
  _gradBall.position.set(x, y, z);
}

let _gradInterval;
function toggleGradPlay() {
  const btn = document.getElementById('grad-play-btn');
  if (_gradPlaying) {
    clearInterval(_gradInterval);
    _gradPlaying = false;
    btn.textContent = '▶ Play Animation';
  } else {
    _gradPlaying = true;
    btn.textContent = '⏸ Pause';
    _gradInterval = setInterval(() => {
      if (_gradStep >= (_gradPath.length - 1)) {
        clearInterval(_gradInterval);
        _gradPlaying = false;
        btn.textContent = '▶ Play Animation';
        return;
      }
      _gradStep++;
      moveBallTo(_gradPath[_gradStep]);
      updateGradInfo(_gradStep);
    }, 80);
  }
}

function resetGrad() {
  clearInterval(_gradInterval);
  _gradPlaying = false;
  _gradStep = 0;
  document.getElementById('grad-play-btn').textContent = '▶ Play Animation';
  if (_gradPath.length) moveBallTo(_gradPath[0]);
  updateGradInfo(0);
}

function updateGradInfo(step) {
  const pt = _gradPath[step];
  if (!pt) return;
  document.getElementById('grad-step').textContent = step;
  document.getElementById('grad-loss').textContent  = pt.loss.toFixed(6);
  document.getElementById('grad-w1').textContent    = pt.w1.toFixed(4);
  document.getElementById('grad-w2').textContent    = pt.w2.toFixed(4);
}

function buildGradLossChart(path) {
  const canvas = document.getElementById('grad-loss-chart');
  if (_gradLossChart) { _gradLossChart.destroy(); _gradLossChart = null; }

  _gradLossChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: path.map(p => p.step),
      datasets: [{
        label: 'Loss',
        data: path.map(p => p.loss),
        borderColor: '#fbbf24',
        backgroundColor: 'rgba(251,191,36,0.1)',
        borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3
      }]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#475569', font: { size: 9 } }, grid: { color: '#1e2d45' } },
        y: { ticks: { color: '#475569', font: { size: 9 } }, grid: { color: '#1e2d45' } }
      }
    }
  });
}

function viridisRGB(t) {
  t = Math.max(0, Math.min(1, t));
  const r = [68,72,67,56,45,37,30,43,81,132,186,253];
  const g = [1,40,90,125,155,184,211,229,243,253,222,231];
  const b = [84,115,140,140,130,121,102,88,73,37,30,37];
  const i = Math.min(10, Math.floor(t * 11));
  const f = t * 11 - i;
  return [
    (r[i] + (r[i+1]-r[i])*f) / 255,
    (g[i] + (g[i+1]-g[i])*f) / 255,
    (b[i] + (b[i+1]-b[i])*f) / 255
  ];
}
