// Pose inference worker: keeps the ML model off the render thread so the 3D
// scene never waits on inference. Protocol:
//   in : {type:'init', engine, wasmBase, modelUrl, modelName, fallbackModelUrl?, probeUrl?, delegate}
//        {type:'frame', bitmap:ImageBitmap, ts:number, id:number}
//   out: {type:'ready', delegate, model, bench} | {type:'error', message}
//        {type:'result', id, ts, inferMs, world?, norm?}
//        {type:'fatal', message}   inference keeps failing → main restarts us
//
// Device auto-tuning happens ONCE, during init (the loading screen), on a
// probe photo of a person: every model × backend combination is timed and the
// best is kept. Nothing is swapped mid-fight, so tracking never hiccups later.
import { PoseEngine } from './PoseEngine.js';

const MAX_MS = 28; // above this per-frame cost, prefer the lighter model
const FATAL_AFTER = 8; // consecutive inference failures before asking for a restart

let engine = null;
let failures = 0;
let probeTs = 1;

const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];

async function loadProbe(url) {
  const blob = await (await fetch(url)).blob();
  const bmp = await createImageBitmap(blob);
  const w = 384;
  const h = Math.round((w * bmp.height) / bmp.width);
  const small = await createImageBitmap(bmp, { resizeWidth: w, resizeHeight: h });
  bmp.close();
  return small;
}

function timeEngine(e, probe) {
  for (let i = 0; i < 3; i++) e.detect(probe, probeTs++); // warm-up (shader/graph init)
  const t = [];
  for (let i = 0; i < 7; i++) {
    const t0 = performance.now();
    e.detect(probe, probeTs++);
    t.push(performance.now() - t0);
  }
  return median(t);
}

async function tuneMediaPipe(msg) {
  const probe = msg.probeUrl ? await loadProbe(msg.probeUrl).catch(() => null) : null;
  if (!probe) {
    // no probe → plain init (GPU with automatic CPU fallback)
    const e = new PoseEngine();
    const d = await e.init(msg);
    return { engine: e, delegate: d, model: msg.modelName, bench: null };
  }
  const models = [{ name: msg.modelName, url: msg.modelUrl }];
  if (msg.fallbackModelUrl) models.push({ name: 'lite', url: msg.fallbackModelUrl });
  const bench = {};
  let best = null;
  for (const mdl of models) {
    for (const d of msg.forceCPU ? ['CPU'] : ['GPU', 'CPU']) {
      const e = new PoseEngine();
      try {
        const got = await e.init({ ...msg, modelUrl: mdl.url, delegate: d });
        if (got !== d) {
          e.close();
          continue;
        }
        const ms = timeEngine(e, probe);
        bench[`${mdl.name}/${d}`] = Math.round(ms * 10) / 10;
        if (!best || ms < best.ms) {
          best?.engine.close();
          best = { engine: e, ms, model: mdl.name, delegate: d };
        } else {
          e.close();
        }
      } catch (err) {
        bench[`${mdl.name}/${d}`] = 'failed';
        e.close();
      }
    }
    // prefer the more accurate model whenever it's fast enough
    if (best && best.ms <= MAX_MS) break;
  }
  probe.close();
  if (!best) throw new Error('No pose backend could start');
  return { engine: best.engine, delegate: best.delegate, model: best.model, bench };
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'init') {
    try {
      if (msg.engine === 'movenet') {
        // MoveNet benchmarks its own backends (webgl vs wasm) during init
        const { MoveNetEngine } = await import('./MoveNetEngine.js');
        engine = new MoveNetEngine();
        const delegate = await engine.init(msg);
        self.postMessage({ type: 'ready', delegate: `movenet/${delegate}`, model: 'movenet', bench: engine.timings });
        return;
      }
      const t = await tuneMediaPipe(msg);
      engine = t.engine;
      self.postMessage({ type: 'ready', delegate: t.delegate, model: t.model, bench: t.bench });
    } catch (err) {
      self.postMessage({ type: 'error', message: String(err?.message || err) });
    }
    return;
  }

  if (msg.type === 'frame') {
    if (!engine) {
      msg.bitmap.close();
      self.postMessage({ type: 'result', id: msg.id, ts: msg.ts, inferMs: 0, skipped: true });
      return;
    }
    const t0 = performance.now();
    let out = null;
    try {
      out = engine.detect(msg.bitmap, msg.ts);
      failures = 0;
    } catch (err) {
      failures++;
      console.error('[pose worker]', err);
      if (failures >= FATAL_AFTER) self.postMessage({ type: 'fatal', message: String(err?.message || err) });
    } finally {
      msg.bitmap.close();
    }
    const inferMs = performance.now() - t0;
    // ALWAYS answer, even on failure — the main thread waits for this reply
    if (out) {
      const transfer = out.world ? [out.world.buffer, out.norm.buffer] : [out.norm.buffer];
      self.postMessage({ type: 'result', id: msg.id, ts: msg.ts, inferMs, world: out.world, norm: out.norm }, transfer);
    } else {
      self.postMessage({ type: 'result', id: msg.id, ts: msg.ts, inferMs });
    }
  }
};
