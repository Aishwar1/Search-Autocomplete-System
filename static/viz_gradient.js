/* Gradient Descent — Three.js 3D Loss Surface + Animated Ball */

let _gradScene, _gradCamera, _gradRenderer, _gradAnimId;
let _gradData = null;
let _gradStep = 0;
let _gradPlaying = false;
let _gradBall = null;
let _gradPath = [];
let _gradLossChart = null;
let _gradInterval;

// Data-to-scene coordinate mapping (set when surface is loaded)
let _gradW1Min = -2, _gradW1Max = 2;  // default fallback
let _gradW2Min = -1, _gradW2Max = 3;  // default fallback — actual backend range

function dataToScene(w1, w2, loss) {
  // Map data ranges to scene [-2, 2] for x and z
  const x = ((w1 - _gradW1Min) / (_gradW1Max - _gradW1Min)) * 4 - 2;
  const z = ((w2 - _gradW2Min) / (_gradW2Max - _gradW2Min)) * 4 - 2;
  const y = Math.min(loss * 0.4, 3);
  return { x, y, z };
}

function initGradientViz() {
  if (typeof THREE === 'undefined') return;

  const container = document.getElementById('grad-3d-container');
  const W = container.clientWidth || 640;
  const H = 500;

  _gradScene = new THREE.Scene();
  _gradScene.background = new THREE.Color(0x000000);

  _gradCamera = new THREE.PerspectiveCamera(55, W / H, 0.01, 200);
  _gradCamera.position.set(5, 6, 8);
  _gradCamera.lookAt(0, 0, 0);

  _gradRenderer = new THREE.WebGLRenderer({ antialias: false });
  _gradRenderer.setSize(W, H);
  container.appendChild(_gradRenderer.domElement);

  // Lighting
  _gradScene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 0.7);
  dir.position.set(5, 10, 5);
  _gradScene.add(dir);

  // Manual orbit
  let isDrag = false, prevMouse = { x: 0, y: 0 };
  let sph = { theta: 0.7, phi: 0.9, r: 10 };

  function updateCam() {
    _gradCamera.position.x = sph.r * Math.sin(sph.phi) * Math.sin(sph.theta);
    _gradCamera.position.y = sph.r * Math.cos(sph.phi);
    _gradCamera.position.z = sph.r * Math.sin(sph.phi) * Math.cos(sph.theta);
    _gradCamera.lookAt(0, 0, 0);
  }
  updateCam();

  _gradRenderer.domElement.addEventListener('mousedown', e => {
    isDrag = true; prevMouse = { x: e.clientX, y: e.clientY };
  });
  window.addEventListener('mouseup', () => { isDrag = false; });
  window.addEventListener('mousemove', e => {
    if (!isDrag) return;
    sph.theta -= (e.clientX - prevMouse.x) * 0.009;
    sph.phi   -= (e.clientY - prevMouse.y) * 0.009;
    sph.phi = Math.max(0.15, Math.min(Math.PI * 0.45, sph.phi));
    prevMouse = { x: e.clientX, y: e.clientY };
    updateCam();
  });
  _gradRenderer.domElement.addEventListener('wheel', e => {
    sph.r = Math.max(4, Math.min(20, sph.r + e.deltaY * 0.02));
    updateCam();
    e.preventDefault();
  }, { passive: false });

  function animate() {
    _gradAnimId = requestAnimationFrame(animate);
    _gradRenderer.render(_gradScene, _gradCamera);
  }
  animate();

  window.addEventListener('resize', () => {
    const W2 = container.clientWidth;
    _gradCamera.aspect = W2 / H;
    _gradCamera.updateProjectionMatrix();
    _gradRenderer.setSize(W2, H);
  });

  document.getElementById('grad-play-btn').addEventListener('click', toggleGradPlay);
  document.getElementById('grad-reset-btn').addEventListener('click', resetGrad);
}

async function loadGradientSurface() {
  try {
    const res = await fetch('/api/gradient');
    _gradData = await res.json();

    // Capture actual data ranges from the API so coordinate transforms are consistent
    const w1Arr = _gradData.w1 || [];
    const w2Arr = _gradData.w2 || [];
    if (w1Arr.length) { _gradW1Min = w1Arr[0]; _gradW1Max = w1Arr[w1Arr.length - 1]; }
    if (w2Arr.length) { _gradW2Min = w2Arr[0]; _gradW2Max = w2Arr[w2Arr.length - 1]; }

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

  const vertices = [];
  const colors   = [];
  const indices  = [];

  let lMin = Infinity, lMax = -Infinity;
  surface.forEach(row => row.forEach(v => {
    lMin = Math.min(lMin, v);
    lMax = Math.max(lMax, v);
  }));

  for (let j = 0; j < M; j++) {
    for (let i = 0; i < N; i++) {
      const wv1 = w1[i] !== undefined ? w1[i] : ((i / (N - 1)) * (_gradW1Max - _gradW1Min) + _gradW1Min);
      const wv2 = w2[j] !== undefined ? w2[j] : ((j / (M - 1)) * (_gradW2Max - _gradW2Min) + _gradW2Min);
      const loss = surface[j] ? (surface[j][i] || 0) : 0;
      const { x, y, z } = dataToScene(wv1, wv2, loss);

      vertices.push(x, y, z);

      // Color: dark cool for low loss, brighter warm for high
      const t = (loss - lMin) / (lMax - lMin + 1e-10);
      const rgb = gradViridisRGB(t);
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
    roughness: 0.9, metalness: 0.0
  });

  const mesh = new THREE.Mesh(geo, mat);
  _gradScene.add(mesh);

  // Wireframe overlay (very subtle)
  const wMat = new THREE.MeshBasicMaterial({
    wireframe: true, color: 0x1a1a1a, opacity: 0.2, transparent: true
  });
  _gradScene.add(new THREE.Mesh(geo, wMat));

  // Minimum marker — placed at the actual Rosenbrock minimum (w1=1, w2=1)
  const minScene = dataToScene(1.0, 1.0, 0.0);
  const minGeo = new THREE.ConeGeometry(0.1, 0.35, 8);
  const minMat = new THREE.MeshStandardMaterial({ color: 0x52d18a });
  const minMarker = new THREE.Mesh(minGeo, minMat);
  minMarker.position.set(minScene.x, minScene.y + 0.18, minScene.z);
  _gradScene.add(minMarker);

  // Axis lines
  const axesMat = (c) => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.4 });
  [
    { pts: [[-2,0,0],[2,0,0]], c: 0xe05555 },
    { pts: [[0,0,0],[0,3,0]], c: 0x52d18a },
    { pts: [[0,0,-2],[0,0,2]], c: 0x5a9ae8 }
  ].forEach(ax => {
    const geo2 = new THREE.BufferGeometry().setFromPoints(ax.pts.map(p => new THREE.Vector3(...p)));
    _gradScene.add(new THREE.Line(geo2, axesMat(ax.c)));
  });
}

function buildGradPath(pathData) {
  if (!_gradScene || !pathData.length) return;
  _gradPath = pathData;

  const ballGeo = new THREE.SphereGeometry(0.1, 10, 10);
  const ballMat = new THREE.MeshStandardMaterial({ color: 0xd4a843, roughness: 0.4 });
  _gradBall = new THREE.Mesh(ballGeo, ballMat);
  _gradScene.add(_gradBall);

  moveBallTo(pathData[0]);
}

function moveBallTo(pt) {
  if (!_gradBall || !pt) return;
  // Use the same data→scene mapping as the surface mesh
  const { x, y, z } = dataToScene(pt.w1, pt.w2, pt.loss);
  _gradBall.position.set(x, y + 0.12, z);
}

function toggleGradPlay() {
  const btn = document.getElementById('grad-play-btn');
  if (_gradPlaying) {
    clearInterval(_gradInterval);
    _gradPlaying = false;
    btn.textContent = 'Play Animation';
  } else {
    _gradPlaying = true;
    btn.textContent = 'Pause';
    _gradInterval = setInterval(() => {
      if (_gradStep >= (_gradPath.length - 1)) {
        clearInterval(_gradInterval);
        _gradPlaying = false;
        btn.textContent = 'Play Animation';
        return;
      }
      _gradStep++;
      moveBallTo(_gradPath[_gradStep]);
      updateGradInfo(_gradStep);
    }, 60);
  }
}

function resetGrad() {
  clearInterval(_gradInterval);
  _gradPlaying = false;
  _gradStep = 0;
  document.getElementById('grad-play-btn').textContent = 'Play Animation';
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
        borderColor: '#d4a843',
        backgroundColor: 'rgba(212,168,67,0.07)',
        borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.3
      }]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#444', font: { size: 9 } }, grid: { color: '#1a1a1a' } },
        y: { ticks: { color: '#444', font: { size: 9 } }, grid: { color: '#1a1a1a' } }
      }
    }
  });
}

function gradViridisRGB(t) {
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
