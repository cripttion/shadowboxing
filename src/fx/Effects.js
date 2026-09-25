// Impact particles (one pooled Points draw call), camera shake and hit-stop.
import * as THREE from 'three';

const MAX = 400;

export class Effects {
  constructor(scene) {
    this.pos = new Float32Array(MAX * 3);
    this.vel = new Float32Array(MAX * 3);
    this.life = new Float32Array(MAX);
    this.size = new Float32Array(MAX);
    this.color = new Float32Array(MAX * 3);
    this.next = 0;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    g.setAttribute('aLife', new THREE.BufferAttribute(this.life, 1));
    g.setAttribute('aSize', new THREE.BufferAttribute(this.size, 1));
    g.setAttribute('aColor', new THREE.BufferAttribute(this.color, 3));
    const m = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */ `
        attribute float aLife; attribute float aSize; attribute vec3 aColor;
        varying float vL; varying vec3 vC;
        void main(){ vL = aLife; vC = aColor;
          vec4 mv = modelViewMatrix*vec4(position,1.0);
          gl_PointSize = aSize * (0.4 + aLife) * 300.0 / -mv.z;
          gl_Position = projectionMatrix*mv; }`,
      fragmentShader: /* glsl */ `
        varying float vL; varying vec3 vC;
        void main(){ if (vL <= 0.0) discard; float d = length(gl_PointCoord-0.5);
          float a = smoothstep(0.5, 0.0, d);
          gl_FragColor = vec4(vC * a * vL, 1.0); }`,
    });
    this.points = new THREE.Points(g, m);
    this.points.frustumCulled = false;
    scene.add(this.points);

    // Flash sprite at the impact point
    this.flash = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: flashTexture(), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }),
    );
    this.flash.scale.setScalar(0.5);
    scene.add(this.flash);
    this.flashLife = 0;

    this.shake = 0;
    this.shakeOffset = new THREE.Vector3();
    this.hitStop = 0;
    this.slowMo = 0;
  }

  burst(p, dir, { count = 26, speed = 3, color = [1, 0.8, 0.5], size = 0.05, spread = 1 } = {}) {
    for (let i = 0; i < count; i++) {
      const k = this.next;
      this.next = (this.next + 1) % MAX;
      this.pos[k * 3] = p.x;
      this.pos[k * 3 + 1] = p.y;
      this.pos[k * 3 + 2] = p.z;
      const s = speed * (0.3 + Math.random());
      this.vel[k * 3] = (dir.x + (Math.random() - 0.5) * spread) * s;
      this.vel[k * 3 + 1] = (dir.y + (Math.random() - 0.2) * spread) * s;
      this.vel[k * 3 + 2] = (dir.z + (Math.random() - 0.5) * spread) * s;
      this.life[k] = 0.6 + Math.random() * 0.4;
      this.size[k] = size * (0.5 + Math.random());
      this.color.set(color, k * 3);
    }
  }

  impact(p, dir, power, blocked) {
    if (blocked) {
      this.burst(p, dir, { count: 14, speed: 1.6, color: [0.6, 0.75, 1], size: 0.04 });
    } else {
      // sweat spray + sparks
      this.burst(p, dir, { count: 18 + power * 16, speed: 2.5 + power * 1.5, color: [0.85, 0.9, 1], size: 0.03 });
      this.burst(p, dir, { count: 10, speed: 1.2, color: [1, 0.55, 0.25], size: 0.08, spread: 1.6 });
    }
    this.flash.position.copy(p);
    this.flash.material.color.set(blocked ? 0x88aaff : 0xffe2b0);
    this.flashLife = blocked ? 0.5 : 1;
    this.shake = Math.max(this.shake, blocked ? 0.12 : 0.25 + power * 0.25);
    this.hitStop = blocked ? 0.02 : 0.045 + power * 0.03;
  }

  update(dt) {
    for (let i = 0; i < MAX; i++) {
      if (this.life[i] <= 0) continue;
      this.life[i] -= dt * 1.8;
      this.vel[i * 3 + 1] -= 6 * dt;
      this.pos[i * 3] += this.vel[i * 3] * dt;
      this.pos[i * 3 + 1] += this.vel[i * 3 + 1] * dt;
      this.pos[i * 3 + 2] += this.vel[i * 3 + 2] * dt;
    }
    const a = this.points.geometry.attributes;
    a.position.needsUpdate = a.aLife.needsUpdate = a.aSize.needsUpdate = a.aColor.needsUpdate = true;

    this.flashLife = Math.max(0, this.flashLife - dt * 7);
    this.flash.material.opacity = this.flashLife;
    this.flash.scale.setScalar(0.25 + (1 - this.flashLife) * 0.5);

    this.shake = Math.max(0, this.shake - dt * 2.2);
    const s = this.shake * this.shake * 0.06;
    const t = performance.now() * 0.05;
    this.shakeOffset.set(Math.sin(t * 1.3) * s, Math.sin(t * 1.7 + 1) * s, 0);
  }
}

function flashTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.25, 'rgba(255,230,190,0.7)');
  grd.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  // star streaks
  g.globalCompositeOperation = 'lighter';
  g.strokeStyle = 'rgba(255,240,220,0.8)';
  g.lineWidth = 3;
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    g.beginPath();
    g.moveTo(64, 64);
    g.lineTo(64 + Math.cos(a) * 62, 64 + Math.sin(a) * 62);
    g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
