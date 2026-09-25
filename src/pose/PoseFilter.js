// Jitter removal + latency compensation for landmark streams.
//
// One Euro filter (Casiez et al. 2012): heavy smoothing when a joint is still
// (kills camera jitter), almost none when it moves fast (punches stay snappy).
// On top of that, every render frame extrapolates each joint along its
// velocity by the age of the last camera frame, so the avatar shows where your
// fist *is now*, not where it was when the sensor captured it.
import { NUM_LANDMARKS, STRIDE } from './landmarks.js';

const TWO_PI = Math.PI * 2;
const alpha = (cutoff, dt) => {
  const tau = 1 / (TWO_PI * cutoff);
  return 1 / (1 + tau / dt);
};

class OneEuroBank {
  constructor(count, { minCutoff, beta, dCutoff = 1.0, velCutoff = 8 }) {
    this.n = count;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.velCutoff = velCutoff;
    this.x = new Float32Array(count);
    this.dx = new Float32Array(count);
    this.vel = new Float32Array(count); // smoothed velocity used for prediction
    this.init = false;
  }

  reset() {
    this.init = false;
  }

  update(values, dt) {
    const { n, x, dx, vel } = this;
    if (!this.init || dt <= 0) {
      for (let i = 0; i < n; i++) {
        x[i] = values[i];
        dx[i] = 0;
        vel[i] = 0;
      }
      this.init = true;
      return;
    }
    const ad = alpha(this.dCutoff, dt);
    const av = alpha(this.velCutoff, dt);
    for (let i = 0; i < n; i++) {
      const raw = values[i];
      const d = (raw - x[i]) / dt;
      dx[i] += ad * (d - dx[i]);
      const cutoff = this.minCutoff + this.beta * Math.abs(dx[i]);
      const a = alpha(cutoff, dt);
      const prev = x[i];
      x[i] += a * (raw - x[i]);
      vel[i] += av * ((x[i] - prev) / dt - vel[i]);
    }
  }
}

/**
 * Filters world (metric, 3D) and normalized (image, 2D) landmarks and exposes
 * a predicted sample at arbitrary render time.
 */
export class PoseFilter {
  constructor() {
    const N = NUM_LANDMARKS;
    // dCutoff is how fast the filter notices motion starting: high values open
    // the filter almost instantly when a punch begins (low onset lag).
    this.world = new OneEuroBank(N * 3, { minCutoff: 1.7, beta: 9.0, dCutoff: 4.0 });
    this.norm = new OneEuroBank(N * 2, { minCutoff: 1.5, beta: 16.0, dCutoff: 4.0 });
    this.vis = new Float32Array(N);
    this._rawW = new Float32Array(N * 3);
    this._rawN = new Float32Array(N * 2);
    this.lastTs = 0;
    this.hasData = false;
    this.lostAt = 0;
    // Output buffers (reused every frame — no GC pressure in the hot loop)
    this.outWorld = new Float32Array(N * 3);
    this.outNorm = new Float32Array(N * 2);
    this.maxLead = 0.06; // seconds of extrapolation we allow at most
    this.extraLead = 0.012; // compensates the display pipeline (~1 frame)
    this.maxOffset = 0.08; // metres — clamp prediction overshoot
  }

  push(frame) {
    if (!frame) {
      if (this.hasData && !this.lostAt) this.lostAt = performance.now();
      return;
    }
    this.lostAt = 0;
    const { world, norm, ts } = frame;
    const N = NUM_LANDMARKS;
    for (let i = 0; i < N; i++) {
      const o = i * STRIDE;
      if (world) {
        this._rawW[i * 3] = world[o];
        this._rawW[i * 3 + 1] = world[o + 1];
        this._rawW[i * 3 + 2] = world[o + 2];
      }
      this._rawN[i * 2] = norm[o];
      this._rawN[i * 2 + 1] = norm[o + 1];
      // visibility: fast attack, slower release
      const v = norm[o + 3];
      this.vis[i] += (v - this.vis[i]) * (v > this.vis[i] ? 0.6 : 0.25);
    }
    const dt = this.lastTs ? (ts - this.lastTs) / 1000 : 0;
    if (dt > 0.5) {
      this.world.reset();
      this.norm.reset();
    }
    // 2D-only trackers (MoveNet) have no metric world landmarks
    this.hasWorld = !!world;
    if (world) this.world.update(this._rawW, Math.min(dt, 0.25));
    this.norm.update(this._rawN, Math.min(dt, 0.25));
    this.lastTs = ts;
    this.hasData = true;
  }

  /** True when tracking was lost for more than `ms`. */
  isLost(ms = 400) {
    return !this.hasData || (this.lostAt && performance.now() - this.lostAt > ms);
  }

  /** Predict landmarks at render time `now` (performance.now() ms). */
  sample(now) {
    const lead = Math.min(this.maxLead, Math.max(0, (now - this.lastTs) / 1000 + this.extraLead));
    const W = this.world;
    const Nn = this.norm;
    const mo = this.maxOffset;
    for (let i = 0; i < W.n; i++) {
      let off = W.vel[i] * lead;
      if (off > mo) off = mo;
      else if (off < -mo) off = -mo;
      this.outWorld[i] = W.x[i] + off;
    }
    // image-space prediction (clamped to 4% of the frame to avoid overshoot)
    for (let i = 0; i < Nn.n; i++) {
      let off = Nn.vel[i] * lead;
      if (off > 0.04) off = 0.04;
      else if (off < -0.04) off = -0.04;
      this.outNorm[i] = Nn.x[i] + off;
    }
    return { world: this.hasWorld ? this.outWorld : null, norm: this.outNorm, vis: this.vis };
  }
}
