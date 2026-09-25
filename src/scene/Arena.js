// Boxing arena: ring, ropes, corner posts, light rig with volumetric beams,
// animated crowd and dust. Built procedurally (tiny download) with draw calls
// kept low through merging and instancing.
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';

const RING = 3.2; // half size of the canvas (m)
const APRON = 1.05; // platform height

export class Arena {
  constructor(scene, renderer, preset) {
    this.scene = scene;
    this.preset = preset;
    this.renderer = renderer;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.time = 0;

    scene.background = new THREE.Color(0x05060a);
    scene.fog = new THREE.FogExp2(0x06070c, 0.045);

    this._lights();
    this._ring();
    this._truss();
    this._crowd();
    this._dust();
  }

  async loadEnvironment(url) {
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const hdr = await new HDRLoader().loadAsync(url);
    const env = pmrem.fromEquirectangular(hdr).texture;
    hdr.dispose();
    pmrem.dispose();
    this.scene.environment = env;
    this.scene.environmentIntensity = 0.42;
  }

  _lights() {
    const s = this.scene;
    s.add(new THREE.HemisphereLight(0x8aa0ff, 0x1a1010, 0.35));

    // Key: big overhead spot over the centre of the ring
    const key = new THREE.SpotLight(0xfff1dc, 170, 16, 0.62, 0.6, 1.6);
    key.position.set(0.4, 8.5, 1.2);
    key.target.position.set(0, 0.8, 0);
    if (this.preset.shadows) {
      key.castShadow = true;
      key.shadow.mapSize.setScalar(this.preset.shadowSize);
      key.shadow.bias = -0.0004;
      key.shadow.normalBias = 0.02;
      key.shadow.camera.near = 3;
      key.shadow.camera.far = 13;
      key.shadow.radius = 4;
    }
    s.add(key, key.target);
    this.key = key;

    // Coloured rims: blue behind the player, red behind the opponent
    const rimA = new THREE.DirectionalLight(0x4f7dff, 1.6);
    rimA.position.set(-4, 3.2, 5);
    const rimB = new THREE.DirectionalLight(0xff4a3a, 1.5);
    rimB.position.set(4.5, 3.4, -5);
    const front = new THREE.DirectionalLight(0xffffff, 0.45);
    front.position.set(1.5, 2.5, 6);
    s.add(rimA, rimB, front);
  }

  _canvasTexture() {
    const c = document.createElement('canvas');
    c.width = c.height = 1024;
    const g = c.getContext('2d');
    const grd = g.createRadialGradient(512, 512, 80, 512, 512, 720);
    grd.addColorStop(0, '#dfe4ec');
    grd.addColorStop(1, '#aab3c2');
    g.fillStyle = grd;
    g.fillRect(0, 0, 1024, 1024);
    // canvas weave noise
    const img = g.getImageData(0, 0, 1024, 1024);
    for (let i = 0; i < img.data.length; i += 4) {
      const n = (Math.random() - 0.5) * 14 + ((i / 4) % 2 ? 3 : -3);
      img.data[i] += n;
      img.data[i + 1] += n;
      img.data[i + 2] += n;
    }
    g.putImageData(img, 0, 0);
    // scuffs
    for (let i = 0; i < 60; i++) {
      g.fillStyle = `rgba(60,55,50,${Math.random() * 0.05})`;
      g.beginPath();
      g.ellipse(Math.random() * 1024, Math.random() * 1024, 10 + Math.random() * 60, 6 + Math.random() * 30, Math.random() * 3, 0, Math.PI * 2);
      g.fill();
    }
    // centre logo
    g.save();
    g.translate(512, 512);
    g.strokeStyle = 'rgba(180,20,30,0.85)';
    g.lineWidth = 14;
    g.beginPath();
    g.arc(0, 0, 230, 0, Math.PI * 2);
    g.stroke();
    g.lineWidth = 4;
    g.beginPath();
    g.arc(0, 0, 205, 0, Math.PI * 2);
    g.stroke();
    g.fillStyle = 'rgba(20,24,40,0.88)';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.font = '900 108px "Anton", Impact, "Arial Black", sans-serif';
    g.fillText('SHADOW', 0, -52);
    g.fillStyle = 'rgba(180,20,30,0.9)';
    g.fillText('BOXING', 0, 62);
    g.restore();
    // border stripe
    g.strokeStyle = 'rgba(25,35,70,0.9)';
    g.lineWidth = 26;
    g.strokeRect(20, 20, 984, 984);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    return tex;
  }

  _apronTexture() {
    const c = document.createElement('canvas');
    c.width = 1024;
    c.height = 256;
    const g = c.getContext('2d');
    const grd = g.createLinearGradient(0, 0, 0, 256);
    grd.addColorStop(0, '#141a33');
    grd.addColorStop(1, '#070912');
    g.fillStyle = grd;
    g.fillRect(0, 0, 1024, 256);
    g.fillStyle = '#c8102e';
    g.fillRect(0, 0, 1024, 14);
    g.font = '900 110px "Anton", Impact, sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillStyle = 'rgba(255,255,255,0.92)';
    g.fillText('SHADOW  BOXING  ARENA', 512, 136);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  _ring() {
    const G = this.group;
    // Canvas
    const mat = new THREE.MeshStandardMaterial({ map: this._canvasTexture(), roughness: 0.92, metalness: 0 });
    const floor = new THREE.Mesh(new THREE.BoxGeometry(RING * 2 + 0.6, 0.1, RING * 2 + 0.6), mat);
    floor.position.y = -0.05;
    floor.receiveShadow = true;
    G.add(floor);

    // Apron skirt with branding
    const skirtMat = new THREE.MeshStandardMaterial({ map: this._apronTexture(), roughness: 0.8 });
    const skirtGeo = [];
    for (let i = 0; i < 4; i++) {
      const p = new THREE.PlaneGeometry(RING * 2 + 0.6, APRON);
      p.translate(0, -APRON / 2 - 0.1, RING + 0.3);
      p.rotateY((i * Math.PI) / 2);
      skirtGeo.push(p);
    }
    G.add(new THREE.Mesh(mergeGeometries(skirtGeo), skirtMat));

    // Arena floor
    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(40, 48),
      new THREE.MeshStandardMaterial({ color: 0x0b0c10, roughness: 0.55, metalness: 0.2 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -APRON - 0.1;
    G.add(ground);

    // Posts
    const postGeo = [];
    const padGeo = { red: [], blue: [], white: [] };
    const corners = [
      [RING, RING, 'blue'],
      [-RING, -RING, 'red'],
      [RING, -RING, 'white'],
      [-RING, RING, 'white'],
    ];
    for (const [x, z, col] of corners) {
      const p = new THREE.CylinderGeometry(0.06, 0.07, 1.5, 16);
      p.translate(x, 0.65, z);
      postGeo.push(p);
      const pad = new THREE.BoxGeometry(0.2, 1.05, 0.2);
      pad.translate(x * 0.985, 0.72, z * 0.985);
      padGeo[col].push(pad);
    }
    G.add(new THREE.Mesh(mergeGeometries(postGeo), new THREE.MeshStandardMaterial({ color: 0xc9ccd4, metalness: 1, roughness: 0.25 })));
    const padMat = (c) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.45 });
    G.add(new THREE.Mesh(mergeGeometries(padGeo.red), padMat(0xb3121f)));
    G.add(new THREE.Mesh(mergeGeometries(padGeo.blue), padMat(0x1537a8)));
    G.add(new THREE.Mesh(mergeGeometries(padGeo.white), padMat(0xe8e8e8)));

    // Ropes: slight sag, three colours
    const heights = [0.42, 0.78, 1.14];
    const colors = [0xc8102e, 0xf2f2f2, 0x1b3fbf];
    const cs = [
      [RING, RING],
      [RING, -RING],
      [-RING, -RING],
      [-RING, RING],
    ];
    heights.forEach((h, hi) => {
      const geos = [];
      for (let i = 0; i < 4; i++) {
        const [ax, az] = cs[i];
        const [bx, bz] = cs[(i + 1) % 4];
        const a = new THREE.Vector3(ax * 0.985, h, az * 0.985);
        const b = new THREE.Vector3(bx * 0.985, h, bz * 0.985);
        const mid = a.clone().lerp(b, 0.5);
        mid.y -= 0.05;
        const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
        geos.push(new THREE.TubeGeometry(curve, 40, 0.028, 8, false));
      }
      const rope = new THREE.Mesh(mergeGeometries(geos), new THREE.MeshStandardMaterial({ color: colors[hi], roughness: 0.4 }));
      rope.castShadow = true;
      G.add(rope);
    });

    // Soft glow pool on the arena floor around the ring
    const glow = new THREE.Mesh(
      new THREE.PlaneGeometry(18, 18),
      new THREE.MeshBasicMaterial({ map: radialTexture('rgba(90,110,200,0.35)'), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
    );
    glow.rotation.x = -Math.PI / 2;
    glow.position.y = -APRON - 0.09;
    G.add(glow);
  }

  _truss() {
    const G = this.group;
    const y = 6.2;
    const S = 3.6;
    // square truss frame
    const bars = [];
    for (let i = 0; i < 4; i++) {
      for (const off of [-0.18, 0.18]) {
        const b = new THREE.BoxGeometry(S * 2 + 0.4, 0.05, 0.05);
        b.translate(0, y + off, S);
        b.rotateY((i * Math.PI) / 2);
        bars.push(b);
      }
    }
    G.add(new THREE.Mesh(mergeGeometries(bars), new THREE.MeshStandardMaterial({ color: 0x2a2d35, metalness: 0.9, roughness: 0.4 })));

    // lamps + volumetric beams
    const lampGeo = new THREE.CylinderGeometry(0.16, 0.2, 0.34, 14);
    const lampMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0xfff4de).multiplyScalar(8) });
    const positions = [];
    for (let i = 0; i < 4; i++) {
      for (const t of [-0.55, 0, 0.55]) {
        const v = new THREE.Vector3(t * S * 2 * 0.5, y - 0.3, S).applyAxisAngle(new THREE.Vector3(0, 1, 0), (i * Math.PI) / 2);
        positions.push(v);
      }
    }
    const lamps = new THREE.InstancedMesh(lampGeo, lampMat, positions.length);
    const m = new THREE.Matrix4();
    positions.forEach((p, i) => lamps.setMatrixAt(i, m.makeTranslation(p.x, p.y, p.z)));
    G.add(lamps);

    const beamMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      uniforms: { uColor: { value: new THREE.Color(0xfff0d8) }, uStrength: { value: 0.1 } },
      vertexShader: /* glsl */ `
        varying float vH; varying vec3 vN; varying vec3 vV;
        void main(){
          vH = uv.y;
          vec4 wp = modelMatrix * instanceMatrix * vec4(position,1.0);
          vN = normalize(mat3(modelMatrix * instanceMatrix) * normal);
          vV = normalize(cameraPosition - wp.xyz);
          gl_Position = projectionMatrix * viewMatrix * wp;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uColor; uniform float uStrength;
        varying float vH; varying vec3 vN; varying vec3 vV;
        void main(){
          // clamp every pow() input: NaNs here would be smeared over the
          // whole frame by the bloom blur
          float ndv = clamp(abs(dot(normalize(vN), normalize(vV))), 0.0, 1.0);
          float edge = ndv * sqrt(ndv);
          float h = clamp(vH, 0.0, 1.0);
          float fade = h * h;
          gl_FragColor = vec4(uColor * uStrength * edge * fade, 1.0);
        }`,
    });
    const beamGeo = new THREE.CylinderGeometry(0.14, 1.25, 6.4, 24, 1, true);
    beamGeo.translate(0, -3.2, 0);
    const beams = new THREE.InstancedMesh(beamGeo, beamMat, positions.length);
    const target = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const down = new THREE.Vector3(0, -1, 0);
    positions.forEach((p, i) => {
      target.set(p.x * 0.18, 0, p.z * 0.18).sub(p).normalize();
      q.setFromUnitVectors(down, target);
      m.compose(p, q, new THREE.Vector3(1, 1, 1));
      beams.setMatrixAt(i, m);
    });
    beams.renderOrder = 10;
    G.add(beams);
  }

  _crowd() {
    const count = this.preset.crowd;
    const tex = crowdTexture();
    const geo = new THREE.PlaneGeometry(0.62, 0.9);
    geo.translate(0, 0.45, 0);
    const phase = new Float32Array(count);
    const tint = new Float32Array(count * 3);
    const mat = new THREE.ShaderMaterial({
      uniforms: { uMap: { value: tex }, uTime: { value: 0 }, uFog: { value: new THREE.Color(0x06070c) } },
      transparent: false,
      vertexShader: /* glsl */ `
        attribute float aPhase; attribute vec3 aTint;
        uniform float uTime;
        varying vec2 vUv; varying vec3 vTint; varying float vDepth;
        void main(){
          vUv = uv; vTint = aTint;
          vec3 p = position;
          float excite = 0.5 + 0.5*sin(uTime*0.7 + aPhase*3.0);
          p.y += abs(sin(uTime*(2.0+aPhase) + aPhase*20.0)) * 0.06 * excite;
          vec4 mv = modelViewMatrix * instanceMatrix * vec4(p,1.0);
          vDepth = -mv.z;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */ `
        uniform sampler2D uMap; uniform vec3 uFog;
        varying vec2 vUv; varying vec3 vTint; varying float vDepth;
        void main(){
          vec4 t = texture2D(uMap, vUv);
          if (t.a < 0.5) discard;
          vec3 c = vTint * t.r * 0.75;
          float f = 1.0 - exp(-vDepth*vDepth*0.006);
          gl_FragColor = vec4(mix(c, uFog, f), 1.0);
          #include <colorspace_fragment>
        }`,
    });
    const crowd = new THREE.InstancedMesh(geo, mat, count);
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const p = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    const palette = [0x2a3350, 0x4a2a2a, 0x2f3a2f, 0x3b3b46, 0x523d2b, 0x1f2a44, 0x552233, 0x444444];
    const col = new THREE.Color();
    for (let i = 0; i < count; i++) {
      // tiered rows on 4 sides
      const side = i % 4;
      const row = Math.floor(Math.random() * 7);
      const along = (Math.random() * 2 - 1) * (7 + row * 0.9);
      const dist = 6.2 + row * 0.95 + Math.random() * 0.2;
      p.set(along, -APRON - 0.1 + row * 0.45, dist).applyAxisAngle(up, (side * Math.PI) / 2);
      q.setFromAxisAngle(up, Math.atan2(-p.x, -p.z)); // face the ring
      const sc = 0.9 + Math.random() * 0.25;
      s.set(sc, sc, sc);
      m.compose(p, q, s);
      crowd.setMatrixAt(i, m);
      phase[i] = Math.random();
      col.setHex(palette[(Math.random() * palette.length) | 0]).multiplyScalar(0.6 + Math.random() * 0.6);
      tint[i * 3] = col.r;
      tint[i * 3 + 1] = col.g;
      tint[i * 3 + 2] = col.b;
    }
    geo.setAttribute('aPhase', new THREE.InstancedBufferAttribute(phase, 1));
    geo.setAttribute('aTint', new THREE.InstancedBufferAttribute(tint, 3));
    crowd.frustumCulled = false;
    this.group.add(crowd);
    this.crowdMat = mat;

    // camera flashes in the stands
    const n = 40;
    const fp = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const side = i % 4;
      const row = Math.random() * 7;
      p.set((Math.random() * 2 - 1) * 9, -APRON + 0.6 + row * 0.45, 6.4 + row * 0.95).applyAxisAngle(up, (side * Math.PI) / 2);
      fp.set([p.x, p.y, p.z], i * 3);
    }
    const fg = new THREE.BufferGeometry();
    fg.setAttribute('position', new THREE.BufferAttribute(fp, 3));
    this.flashLife = new Float32Array(n);
    fg.setAttribute('aLife', new THREE.BufferAttribute(this.flashLife, 1));
    const fm = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */ `
        attribute float aLife; varying float vL;
        void main(){ vL = aLife; vec4 mv = modelViewMatrix*vec4(position,1.0);
          gl_PointSize = 90.0 * aLife / -mv.z; gl_Position = projectionMatrix*mv; }`,
      fragmentShader: /* glsl */ `
        varying float vL;
        void main(){ float d = length(gl_PointCoord-0.5); float a = smoothstep(0.5,0.0,d);
          gl_FragColor = vec4(vec3(1.0,0.97,0.9)*a*a*vL*2.0, 1.0); }`,
    });
    this.flashes = new THREE.Points(fg, fm);
    this.flashes.frustumCulled = false;
    this.group.add(this.flashes);
  }

  _dust() {
    const n = this.preset.dust;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[i * 3] = (Math.random() * 2 - 1) * 3.5;
      pos[i * 3 + 1] = Math.random() * 5.5;
      pos[i * 3 + 2] = (Math.random() * 2 - 1) * 3.5;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const m = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: { uTime: { value: 0 } },
      vertexShader: /* glsl */ `
        uniform float uTime; varying float vA;
        void main(){
          vec3 p = position;
          p.y = mod(p.y + uTime*0.05, 5.5);
          p.x += sin(uTime*0.3 + position.z*2.0)*0.15;
          vec4 mv = modelViewMatrix*vec4(p,1.0);
          vA = smoothstep(0.0, 1.0, p.y) * smoothstep(5.5, 3.5, p.y);
          gl_PointSize = 16.0 / -mv.z; gl_Position = projectionMatrix*mv; }`,
      fragmentShader: /* glsl */ `
        varying float vA;
        void main(){ float d = length(gl_PointCoord-0.5); float a = smoothstep(0.5,0.1,d)*vA*0.35;
          gl_FragColor = vec4(vec3(1.0,0.95,0.85)*a, 1.0); }`,
    });
    this.dust = new THREE.Points(g, m);
    this.dust.frustumCulled = false;
    this.group.add(this.dust);
  }

  /** Crowd reacts: excitement 0..1 bursts camera flashes. */
  cheer(amount = 1) {
    const L = this.flashLife;
    const k = Math.ceil(amount * 8);
    for (let i = 0; i < k; i++) L[(Math.random() * L.length) | 0] = 1;
  }

  update(dt) {
    this.time += dt;
    this.crowdMat.uniforms.uTime.value = this.time;
    this.dust.material.uniforms.uTime.value = this.time;
    const L = this.flashLife;
    for (let i = 0; i < L.length; i++) {
      L[i] = Math.max(0, L[i] - dt * 7);
      if (Math.random() < dt * 0.25) L[i] = 1;
    }
    this.flashes.geometry.attributes.aLife.needsUpdate = true;
  }
}

function radialTexture(color) {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, color);
  grd.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function crowdTexture() {
  // simple human silhouette (head + shoulders + torso) with soft shading
  const c = document.createElement('canvas');
  c.width = 64;
  c.height = 96;
  const g = c.getContext('2d');
  const grd = g.createLinearGradient(0, 0, 64, 0);
  grd.addColorStop(0, '#777');
  grd.addColorStop(0.5, '#fff');
  grd.addColorStop(1, '#666');
  g.fillStyle = grd;
  g.beginPath();
  g.arc(32, 20, 11, 0, Math.PI * 2);
  g.fill();
  g.beginPath();
  g.moveTo(8, 96);
  g.quadraticCurveTo(6, 42, 22, 36);
  g.lineTo(42, 36);
  g.quadraticCurveTo(58, 42, 56, 96);
  g.closePath();
  g.fill();
  const t = new THREE.CanvasTexture(c);
  t.flipY = true;
  return t;
}

export function blobShadow() {
  const m = new THREE.Mesh(
    new THREE.PlaneGeometry(1.1, 1.1),
    new THREE.MeshBasicMaterial({ map: radialTexture('rgba(0,0,0,0.75)'), transparent: true, depthWrite: false, blending: THREE.NormalBlending }),
  );
  m.rotation.x = -Math.PI / 2;
  m.position.y = 0.003;
  m.renderOrder = 1;
  return m;
}
