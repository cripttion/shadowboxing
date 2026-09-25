// WebGL renderer with dynamic resolution: the render scale follows the
// measured frame time so the game holds its frame rate on weak GPUs instead of
// stuttering.
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';

export class Renderer {
  constructor(canvas, tier, preset) {
    this.tier = tier;
    this.preset = preset;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: tier !== 'low',
      powerPreference: 'high-performance',
      stencil: false,
    });
    const r = this.renderer;
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    r.shadowMap.enabled = preset.shadows;
    r.shadowMap.type = THREE.PCFShadowMap;

    this.maxDpr = Math.min(window.devicePixelRatio || 1, preset.maxDpr);
    this.minDpr = Math.min(this.maxDpr, preset.minDpr);
    this.dpr = this.maxDpr;
    this.frameMs = 16.7;
    this._slowFor = 0;
    this._fastFor = 0;
    this.composer = null;
    this.bloom = null;
  }

  setup(scene, camera) {
    this.scene = scene;
    this.camera = camera;
    if (this.preset.bloom) {
      const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
      const target = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, samples: 4 });
      this.composer = new EffectComposer(this.renderer, target);
      this.composer.addPass(new RenderPass(scene, camera));
      this.bloom = new UnrealBloomPass(new THREE.Vector2(size.x / 2, size.y / 2), 0.42, 0.45, 2.2);
      this.composer.addPass(this.bloom);
      this.composer.addPass(new OutputPass());
    }
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.renderer.setPixelRatio(this.dpr);
    this.renderer.setSize(w, h, false);
    if (this.composer) {
      this.composer.setPixelRatio(this.dpr);
      this.composer.setSize(w, h);
    }
    this.camera.aspect = w / h;
    // keep the fighters framed on tall (portrait) screens
    this.camera.fov = w / h < 1 ? 70 : 45;
    this.camera.updateProjectionMatrix();
  }

  /** Adaptive resolution controller. Call once per frame with the real dt. */
  adapt(dtMs) {
    this.frameMs += (dtMs - this.frameMs) * 0.05;
    if (this.frameMs > 21) {
      this._slowFor += dtMs;
      this._fastFor = 0;
    } else if (this.frameMs < 17.5) {
      this._fastFor += dtMs;
      this._slowFor = 0;
    } else {
      this._slowFor = this._fastFor = 0;
    }
    if (this._slowFor > 700 && this.dpr > this.minDpr) {
      this.dpr = Math.max(this.minDpr, this.dpr - 0.1);
      this._slowFor = 0;
      this.resize();
    } else if (this._fastFor > 4000 && this.dpr < this.maxDpr) {
      this.dpr = Math.min(this.maxDpr, this.dpr + 0.1);
      this._fastFor = 0;
      this.resize();
    }
  }

  render() {
    if (this.composer) this.composer.render();
    else this.renderer.render(this.scene, this.camera);
  }
}
