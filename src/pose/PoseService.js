// Camera capture + pose inference scheduler.
//
// Latency design:
//  * requestVideoFrameCallback fires exactly when a new camera frame is ready.
//  * Only ONE frame is ever in flight (back-pressure). If the model is still
//    busy when a new frame arrives we remember that, and the moment the result
//    comes back we send the *newest* frame — no queue ever builds up, so the
//    pose we render is always the freshest one the device can produce.
//  * Frames are downscaled on the GPU (createImageBitmap) and transferred
//    (zero-copy) to a worker, so the render loop never blocks on ML.
//
// Freeze-proofing (a live camera pipeline has many ways to stall):
//  * every send path always clears `busy`, even when grabbing a frame throws
//  * a watchdog resends a frame that got no answer, pumps frames itself if the
//    browser stops firing video-frame callbacks, un-pauses the video, and
//    restarts the worker if it keeps stalling or reports repeated failures
//  * camera loss (another app, OS privacy pause, unplugged) → auto-reconnect
//  * the <video> lives in the document so the browser never throttles it
import { NUM_LANDMARKS, STRIDE } from './landmarks.js';

const INFER_WIDTH = 384;
const STALL_MS = 1500; // a frame unanswered this long is considered lost
const NO_FRAME_MS = 700; // no video-frame callback this long → pump manually

export class PoseService {
  constructor({ model = 'lite', engine = 'mediapipe', onResult, onStatus, onNotice } = {}) {
    this.model = model;
    this.engineName = engine;
    this.onResult = onResult || (() => {});
    this.onStatus = onStatus || (() => {});
    this.onNotice = onNotice || (() => {}); // runtime messages (reconnecting…)
    this.video = null;
    this.stream = null;
    this.worker = null;
    this.engine = null; // main-thread fallback
    this.cfg = null;
    this.busy = false;
    this.pending = false;
    this.sentAt = 0;
    this.frameId = 0;
    this.running = false;
    this.lastFrameTs = 0;
    this.lastVideoCb = 0;
    this.useResize = true;
    this.rvfcHandle = 0;
    this.stallTimes = [];
    this.restarting = false;
    this.stats = { poseHz: 0, inferMs: 0, latencyMs: 0, delegate: '-', mode: '-', stalls: 0, restarts: 0, reconnects: 0 };
    this._lastResultAt = 0;
    this._onVideoFrame = this._onVideoFrame.bind(this);
    this._watchdog = this._watchdog.bind(this);
    this._onVisibility = () => {
      // coming back to the tab: timers were throttled, don't treat it as a stall
      this.sentAt = this.lastVideoCb = performance.now();
    };
  }

  async start() {
    this.onStatus('Requesting camera…');
    this.video = document.createElement('video');
    this.video.playsInline = true;
    this.video.muted = true;
    this.video.autoplay = true;
    // In the document (invisible) so the browser never pauses/throttles it
    Object.assign(this.video.style, {
      position: 'fixed',
      right: '0',
      bottom: '0',
      width: '2px',
      height: '2px',
      opacity: '0.01',
      pointerEvents: 'none',
      zIndex: '-1',
    });
    document.body.appendChild(this.video);
    this.video.addEventListener('pause', () => this.running && this.video.play().catch(() => {}));

    await this._openCamera();

    this.onStatus('Tuning tracking for your device…');
    const base = new URL(import.meta.env.BASE_URL, location.href);
    this.cfg =
      this.engineName === 'movenet'
        ? {
            engine: 'movenet',
            modelUrl: new URL('movenet/model.json', base).href,
            wasmBase: new URL('tfjs-wasm/', base).href,
            delegate: 'auto',
          }
        : {
            engine: 'mediapipe',
            wasmBase: new URL('mediapipe/wasm', base).href,
            modelUrl: new URL(`mediapipe/pose_landmarker_${this.model}.task`, base).href,
            modelName: this.model,
            fallbackModelUrl: this.model !== 'lite' ? new URL('mediapipe/pose_landmarker_lite.task', base).href : null,
            probeUrl: new URL('calib/probe.jpg', base).href,
            delegate: 'GPU',
          };

    try {
      await this._startWorker(this.cfg);
      this.stats.mode = 'worker';
    } catch (err) {
      console.warn('[pose] worker unavailable, running on main thread', err);
      this.worker?.terminate();
      this.worker = null;
      await this._startMainThread();
    }

    this.running = true;
    this.lastVideoCb = this.sentAt = performance.now();
    this._schedule();
    this._watchdogTimer = setInterval(this._watchdog, 250);
    document.addEventListener('visibilitychange', this._onVisibility);
    this.onStatus('');
  }

  async _openCamera() {
    const testSrc = new URLSearchParams(location.search).get('testsrc');
    this.stream = testSrc
      ? await testStream(testSrc)
      : await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 },
            // Higher capture rate = less time a frame waits in the sensor pipeline
            frameRate: { ideal: 60, max: 60 },
          },
        });
    const track = this.stream.getVideoTracks()[0];
    if (track) {
      track.addEventListener('ended', () => this._reconnectCamera('Camera disconnected'));
      track.addEventListener('mute', () => this.onNotice('Camera paused by the system…'));
      track.addEventListener('unmute', () => this.onNotice(''));
    }
    this.video.srcObject = this.stream;
    await this.video.play();
    await new Promise((r) => (this.video.videoWidth ? r() : this.video.addEventListener('loadedmetadata', r, { once: true })));
  }

  async _reconnectCamera(reason) {
    if (!this.running || this._reconnecting) return;
    this._reconnecting = true;
    this.onNotice(`${reason} — reconnecting…`);
    this.stats.reconnects++;
    this.stream?.getTracks().forEach((t) => t.stop());
    for (let attempt = 0; attempt < 20 && this.running; attempt++) {
      try {
        await this._openCamera();
        this.busy = false;
        this._schedule();
        this.onNotice('');
        break;
      } catch (err) {
        await new Promise((r) => setTimeout(r, 1500));
      }
    }
    this._reconnecting = false;
  }

  async _startMainThread() {
    if (this.cfg.engine === 'movenet') {
      const { MoveNetEngine } = await import('./MoveNetEngine.js');
      this.engine = new MoveNetEngine();
    } else {
      const { PoseEngine } = await import('./PoseEngine.js');
      this.engine = new PoseEngine();
    }
    this.stats.delegate = await this.engine.init({ ...this.cfg, useModule: false });
    this.stats.mode = 'main';
  }

  _startWorker(cfg) {
    return new Promise((resolve, reject) => {
      if (typeof OffscreenCanvas === 'undefined' || typeof createImageBitmap === 'undefined') {
        reject(new Error('OffscreenCanvas/createImageBitmap unsupported'));
        return;
      }
      const w = new Worker(new URL('./pose.worker.js', import.meta.url), { type: 'module' });
      const timer = setTimeout(() => reject(new Error('pose worker init timeout')), 45000);
      w.onerror = (e) => {
        clearTimeout(timer);
        reject(e.error || new Error(e.message || 'worker error'));
      };
      w.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'ready') {
          clearTimeout(timer);
          this.stats.delegate = m.delegate;
          this.stats.model = m.model;
          this.stats.bench = m.bench;
          this.worker = w;
          w.onmessage = (ev) => this._onWorkerMessage(ev.data);
          // a crash after init: restart instead of silently freezing
          w.onerror = (ev) => {
            console.error('[pose] worker crashed', ev.message || ev);
            this._restartWorker('worker crashed');
          };
          resolve();
        } else if (m.type === 'error') {
          clearTimeout(timer);
          reject(new Error(m.message));
        }
      };
      w.postMessage({ type: 'init', ...cfg });
    });
  }

  async _restartWorker(reason) {
    if (this.restarting || !this.running) return;
    this.restarting = true;
    this.stats.restarts++;
    console.warn('[pose] restarting tracker:', reason);
    this.onNotice('Tracking hiccup — recovering…');
    this.worker?.terminate();
    this.worker = null;
    this.busy = false;
    try {
      // after repeated trouble, stay on the most robust backend
      const cfg = this.stats.restarts >= 2 ? { ...this.cfg, forceCPU: true } : this.cfg;
      await this._startWorker(cfg);
    } catch (err) {
      console.warn('[pose] worker restart failed, using main thread', err);
      await this._startMainThread().catch((e) => console.error(e));
    }
    this.restarting = false;
    this.stallTimes.length = 0;
    this.sentAt = this.lastVideoCb = performance.now();
    this.onNotice('');
  }

  _schedule() {
    if (!this.running || !this.video) return;
    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
      if (this.rvfcHandle) this.video.cancelVideoFrameCallback?.(this.rvfcHandle);
      this.rvfcHandle = this.video.requestVideoFrameCallback(this._onVideoFrame);
    } else if (!this._polling) {
      // Fallback: poll at display rate, send only when the video time advanced
      this._polling = true;
      let lastT = -1;
      const tick = () => {
        if (!this.running) return;
        if (this.video.currentTime !== lastT) {
          lastT = this.video.currentTime;
          this._onVideoFrame(performance.now(), null, true);
        }
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }
  }

  _onVideoFrame(now, meta, noReschedule) {
    if (!this.running) return;
    if (!noReschedule) this.rvfcHandle = this.video.requestVideoFrameCallback(this._onVideoFrame);
    this.lastVideoCb = performance.now();
    // captureTime (when available) is when the sensor produced the frame — the
    // most honest timestamp for latency compensation.
    this.lastFrameTs = meta?.captureTime || now;
    if (this.busy) {
      this.pending = true;
      return;
    }
    this._send(this.lastFrameTs);
  }

  /** Never throws, never leaves `busy` stuck. */
  async _send(ts) {
    if (this.restarting) return;
    this.busy = true;
    this.pending = false;
    this.sentAt = performance.now();
    const id = ++this.frameId;
    try {
      if (this.worker) {
        const bitmap = await this._grab();
        if (!bitmap || !this.worker) {
          this.busy = false;
          return;
        }
        this.worker.postMessage({ type: 'frame', bitmap, ts, id }, [bitmap]);
      } else if (this.engine) {
        const t0 = performance.now();
        const out = this.engine.detect(this.video, ts);
        this._handleResult({ id, ts, inferMs: performance.now() - t0, world: out?.world, norm: out?.norm });
      } else {
        this.busy = false;
      }
    } catch (err) {
      console.warn('[pose] frame send failed', err);
      this.busy = false;
    }
  }

  async _grab() {
    const v = this.video;
    if (!v || v.readyState < 2 || !v.videoWidth) return null;
    const vw = v.videoWidth;
    const vh = v.videoHeight;
    try {
      return this.useResize && vw > INFER_WIDTH
        ? await createImageBitmap(v, { resizeWidth: INFER_WIDTH, resizeHeight: Math.round((INFER_WIDTH * vh) / vw), resizeQuality: 'low' })
        : await createImageBitmap(v);
    } catch (err) {
      if (this.useResize) {
        this.useResize = false; // some browsers reject resize options
        return null;
      }
      throw err;
    }
  }

  _watchdog() {
    // (runs even when the page reports `hidden`: some embedded browsers and
    // partially covered windows report hidden while still on screen)
    if (!this.running || this.restarting) return;
    const now = performance.now();
    const v = this.video;
    if (v && v.paused) v.play().catch(() => {});
    // 1) a frame went out and never came back
    if (this.busy && now - this.sentAt > STALL_MS) {
      this.stats.stalls++;
      this.stallTimes.push(now);
      this.stallTimes = this.stallTimes.filter((t) => now - t < 10000);
      this.busy = false;
      if (this.stallTimes.length >= 3) {
        this._restartWorker('repeated stalls');
        return;
      }
      this._send(this.lastFrameTs || now);
      return;
    }
    // 2) the browser stopped delivering video-frame callbacks
    if (!this.busy && now - this.lastVideoCb > NO_FRAME_MS && v && v.readyState >= 2) {
      this.lastVideoCb = now;
      this._schedule(); // re-arm the callback…
      this._send(now); // …and pump a frame ourselves meanwhile
    }
    // 3) the camera track died without an 'ended' event
    const track = this.stream?.getVideoTracks()[0];
    if (track && track.readyState === 'ended') this._reconnectCamera('Camera stopped');
  }

  _onWorkerMessage(m) {
    if (m.type === 'result') this._handleResult(m);
    else if (m.type === 'fatal') this._restartWorker(m.message);
  }

  _handleResult(m) {
    const now = performance.now();
    const s = this.stats;
    this.busy = false;
    if (!m.skipped) {
      if (this._lastResultAt) {
        const hz = 1000 / Math.max(1, now - this._lastResultAt);
        s.poseHz += (hz - s.poseHz) * 0.1;
      }
      this._lastResultAt = now;
      s.inferMs += (m.inferMs - s.inferMs) * 0.1;
      s.latencyMs += (now - m.ts - s.latencyMs) * 0.1;
    }
    try {
      if (m.norm) this.onResult({ ts: m.ts, world: m.world, norm: m.norm, aspect: this.aspect });
      else if (!m.skipped) this.onResult(null);
    } catch (err) {
      console.error('[pose] result handler failed', err); // never let a consumer bug stall capture
    }
    if (this.pending && this.running) this._send(this.lastFrameTs);
  }

  get aspect() {
    return this.video && this.video.videoHeight ? this.video.videoWidth / this.video.videoHeight : 4 / 3;
  }

  stop() {
    this.running = false;
    clearInterval(this._watchdogTimer);
    document.removeEventListener('visibilitychange', this._onVisibility);
    this.worker?.terminate();
    this.worker = null;
    this.engine?.close();
    this.engine = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video?.remove();
  }
}

/**
 * Dev aid (?testsrc=/dev/pose.jpg): feeds an image through the exact same
 * capture → worker → filter path as a webcam, gently moving it so the
 * tracker sees motion.
 */
async function testStream(url) {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = url;
  await img.decode();
  const c = document.createElement('canvas');
  c.width = 640;
  c.height = 480;
  const g = c.getContext('2d');
  const draw = (t) => {
    const s = Math.max(c.width / img.width, c.height / img.height);
    const dx = Math.sin(t / 700) * 40;
    g.fillStyle = '#000';
    g.fillRect(0, 0, c.width, c.height);
    g.drawImage(img, (c.width - img.width * s) / 2 + dx, (c.height - img.height * s) / 2, img.width * s, img.height * s);
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
  return c.captureStream(30);
}

export const EMPTY_POSE = () => new Float32Array(NUM_LANDMARKS * STRIDE);
