// MoveNet SinglePose Lightning (TF.js) — a tiny 192×192 model built for
// real-time fitness tracking. It is 2D only; CameraDriver lifts it to 3D
// using bone lengths. Runs inside the pose worker.
//
// Accuracy trick (same idea as Google's reference impl): after the first
// detection we crop a square around the person, so the 192px input is spent
// on the body rather than on the room.
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import { NUM_LANDMARKS, STRIDE } from './landmarks.js';

const SIZE = 192;
// COCO-17 → BlazePose-33 index map, so the rest of the game has one layout
const TO_BLAZE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

export class MoveNetEngine {
  constructor() {
    this.model = null;
    this.delegate = null;
    this.crop = null; // {x, y, size} in source pixels
  }

  /**
   * delegate: 'webgl' | 'wasm' | 'auto'. 'auto' times the real pipeline
   * (bitmap upload + crop + model) on each backend and keeps the fastest —
   * which one wins differs a lot between devices and inside workers.
   */
  async init({ modelUrl, wasmBase, delegate = 'auto' }) {
    if (wasmBase) setWasmPaths(wasmBase.endsWith('/') ? wasmBase : wasmBase + '/');
    const candidates = delegate === 'auto' ? ['webgl', 'wasm'] : [delegate];
    const timings = {};
    let best = null;
    for (const backend of candidates) {
      try {
        const ms = await this._tryBackend(backend, modelUrl, candidates.length > 1);
        timings[backend] = Math.round(ms * 10) / 10;
        if (!best || ms < timings[best]) best = backend;
      } catch (err) {
        console.warn('[movenet] backend failed', backend, err);
      }
    }
    if (!best) throw new Error('No TF.js backend available');
    if (tf.getBackend() !== best) await this._tryBackend(best, modelUrl, false);
    this.delegate = best;
    this.timings = timings;
    return best;
  }

  async _tryBackend(backend, modelUrl, measure) {
    if (backend === 'wasm') await import('@tensorflow/tfjs-backend-wasm');
    const ok = await tf.setBackend(backend);
    if (!ok) throw new Error(`tfjs backend ${backend} unavailable`);
    await tf.ready();
    this.model?.dispose();
    this.model = await loadGraphModel(modelUrl);
    // warm-up compiles shaders / allocates so the first real frame is fast
    tf.tidy(() => this.model.execute(tf.zeros([1, SIZE, SIZE, 3], 'int32'))).dataSync();
    if (!measure) return 0;
    const probe = await makeProbeImage();
    for (let i = 0; i < 3; i++) this.detect(probe);
    const t0 = performance.now();
    const N = 8;
    for (let i = 0; i < N; i++) this.detect(probe);
    this.crop = null;
    return (performance.now() - t0) / N;
  }

  detect(image) {
    const W = image.videoWidth || image.width;
    const H = image.videoHeight || image.height;
    const c = this.crop || fullCrop(W, H);
    const out = tf.tidy(() => {
      // functional ops only: chained tensor methods aren't registered in tfjs-core
      const img = tf.expandDims(tf.cast(tf.browser.fromPixels(image), 'float32'), 0);
      const box = tf.tensor2d([[c.y / H, c.x / W, (c.y + c.size) / H, (c.x + c.size) / W]]);
      // cropAndResize pads outside the image with zeros (letterbox)
      const input = tf.cast(tf.image.cropAndResize(img, box, tf.tensor1d([0], 'int32'), [SIZE, SIZE]), 'int32');
      return this.model.execute(input);
    });
    const kp = out.dataSync(); // 17 × (y, x, score) in crop space
    out.dispose();

    const norm = new Float32Array(NUM_LANDMARKS * STRIDE);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let good = 0;
    let torso = 0;
    for (let i = 0; i < 17; i++) {
      const py = c.y + kp[i * 3] * c.size;
      const px = c.x + kp[i * 3 + 1] * c.size;
      const s = kp[i * 3 + 2];
      const o = TO_BLAZE[i] * STRIDE;
      norm[o] = px / W;
      norm[o + 1] = py / H;
      norm[o + 2] = 0;
      // map MoveNet's conservative scores onto MediaPipe-like visibility
      norm[o + 3] = Math.min(1, Math.max(0, (s - 0.1) / 0.35));
      if (s > 0.25) {
        good++;
        if (i >= 5 && i <= 6) torso++;
        minX = Math.min(minX, px);
        maxX = Math.max(maxX, px);
        minY = Math.min(minY, py);
        maxY = Math.max(maxY, py);
      }
    }

    if (good < 5 || torso < 2) {
      this.crop = null;
      return good < 3 ? null : { norm };
    }
    // next crop: square around the body with room for extended arms
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const size = Math.min(Math.max(W, H) * 1.1, Math.max(maxX - minX, maxY - minY) * 1.7 + 0.15 * Math.min(W, H));
    this.crop = { x: cx - size / 2, y: cy - size / 2, size };
    return { norm };
  }

  close() {
    this.model?.dispose();
    this.model = null;
  }
}

async function makeProbeImage() {
  const c = new OffscreenCanvas(384, 288);
  const g = c.getContext('2d');
  g.fillStyle = '#556';
  g.fillRect(0, 0, 384, 288);
  g.fillStyle = '#c96';
  g.fillRect(170, 60, 44, 200);
  return createImageBitmap(c);
}

function fullCrop(W, H) {
  const size = Math.max(W, H);
  return { x: (W - size) / 2, y: (H - size) / 2, size };
}
