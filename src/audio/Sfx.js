// Fully synthesised sound (WebAudio) — zero audio downloads, near-zero latency.
export class Sfx {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.muted = false;
    this._noise = null;
    this._crowd = null;
  }

  unlock() {
    if (this.ctx) {
      if (this.ctx.state === 'suspended') this.ctx.resume();
      return;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    this.ctx = new AC({ latencyHint: 'interactive' });
    this.master = this.ctx.createGain();
    this.master.gain.value = this.muted ? 0 : 0.8;
    const comp = this.ctx.createDynamicsCompressor();
    comp.threshold.value = -14;
    comp.ratio.value = 4;
    this.master.connect(comp).connect(this.ctx.destination);
    const len = this.ctx.sampleRate * 2;
    const buf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    this._noise = buf;
  }

  setMuted(m) {
    this.muted = m;
    if (this.master) this.master.gain.setTargetAtTime(m ? 0 : 0.8, this.ctx.currentTime, 0.05);
  }

  _noiseSrc(t, dur, type, freq, q, gain) {
    const c = this.ctx;
    gain = Math.max(gain, 0.001); // exponential ramps can't touch 0
    const src = c.createBufferSource();
    src.buffer = this._noise;
    src.playbackRate.value = 0.8 + Math.random() * 0.4;
    const f = c.createBiquadFilter();
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    const g = c.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(f).connect(g).connect(this.master);
    src.start(t, Math.random() * 1.5, dur + 0.05);
    return f;
  }

  _thump(t, f0, f1, dur, gain) {
    const c = this.ctx;
    gain = Math.max(gain, 0.001);
    const o = c.createOscillator();
    o.type = 'sine';
    o.frequency.setValueAtTime(f0, t);
    o.frequency.exponentialRampToValueAtTime(f1, t + dur);
    const g = c.createGain();
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(this.master);
    o.start(t);
    o.stop(t + dur + 0.02);
  }

  punch(power = 1) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this._thump(t, 140 + power * 30, 45, 0.16, 0.9 * power);
    this._noiseSrc(t, 0.09, 'lowpass', 1800 + power * 1200, 0.7, 0.9 * power);
    this._noiseSrc(t, 0.03, 'highpass', 3000, 0.5, 0.25 * power);
  }

  block() {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this._thump(t, 220, 110, 0.08, 0.45);
    this._noiseSrc(t, 0.06, 'bandpass', 900, 1.2, 0.6);
  }

  whoosh(power = 1) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const f = this._noiseSrc(t, 0.2, 'bandpass', 500, 2.5, 0.22 * power);
    f.frequency.exponentialRampToValueAtTime(2400, t + 0.16);
  }

  bell(times = 1) {
    if (!this.ctx) return;
    const c = this.ctx;
    for (let k = 0; k < times; k++) {
      const t = c.currentTime + k * 0.32;
      for (const [ratio, amp] of [[1, 0.5], [2.76, 0.25], [5.4, 0.12], [8.93, 0.06]]) {
        const o = c.createOscillator();
        o.frequency.value = 820 * ratio;
        const g = c.createGain();
        g.gain.setValueAtTime(amp * 0.5, t);
        g.gain.exponentialRampToValueAtTime(0.0001, t + 1.6 / ratio + 0.3);
        o.connect(g).connect(this.master);
        o.start(t);
        o.stop(t + 2);
      }
    }
  }

  crowd(on = true) {
    if (!this.ctx) return;
    if (on && !this._crowd) {
      const c = this.ctx;
      const src = c.createBufferSource();
      src.buffer = this._noise;
      src.loop = true;
      const f = c.createBiquadFilter();
      f.type = 'bandpass';
      f.frequency.value = 700;
      f.Q.value = 0.6;
      const g = c.createGain();
      g.gain.value = 0.045;
      src.connect(f).connect(g).connect(this.master);
      src.start();
      this._crowd = { src, g, f };
    } else if (!on && this._crowd) {
      this._crowd.src.stop();
      this._crowd = null;
    }
  }

  roar(amount = 1) {
    if (!this.ctx || !this._crowd) return;
    const t = this.ctx.currentTime;
    const g = this._crowd.g.gain;
    g.cancelScheduledValues(t);
    g.setValueAtTime(g.value, t);
    g.linearRampToValueAtTime(0.045 + 0.2 * amount, t + 0.15);
    g.exponentialRampToValueAtTime(0.045, t + 1.8);
  }
}
