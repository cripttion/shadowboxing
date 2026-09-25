// DOM HUD. Writes are batched and only touch the DOM when values change, so
// the UI costs next to nothing per frame.
import { SKELETON_EDGES, LM } from '../pose/landmarks.js';

const $ = (s) => document.querySelector(s);

export class Hud {
  constructor() {
    this.root = $('#hud');
    this.youFill = $('.you .bar .fill');
    this.youLag = $('.you .bar .lag');
    this.aiFill = $('.ai .bar .fill');
    this.aiLag = $('.ai .bar .lag');
    this.roundEl = $('.clock .round');
    this.timeEl = $('.clock .time');
    this.lvlEl = $('.ai .lvl');
    this.callouts = $('#callouts');
    this.comboEl = $('#combo');
    this.hintEl = $('#hint');
    this.banner = $('#banner');
    this.damage = $('#damage');
    this.stats = $('#stats');
    this.pip = $('#pip');
    this.pipCanvas = $('#pip canvas');
    this.pipCtx = this.pipCanvas.getContext('2d', { alpha: false });
    this._last = {};
    this._hintTimer = 0;
  }

  show(on) {
    this.root.classList.toggle('hidden', !on);
  }

  setHealth(you, ai) {
    if (this._last.you !== you) {
      this._last.you = you;
      this.youFill.style.transform = `scaleX(${Math.max(0, you) / 100})`;
      this.youLag.style.transform = `scaleX(${Math.max(0, you) / 100})`;
    }
    if (this._last.ai !== ai) {
      this._last.ai = ai;
      this.aiFill.style.transform = `scaleX(${Math.max(0, ai) / 100})`;
      this.aiLag.style.transform = `scaleX(${Math.max(0, ai) / 100})`;
    }
  }

  setClock(round, seconds, totalRounds) {
    let txt = '∞';
    if (Number.isFinite(seconds)) {
      const t = Math.max(0, Math.ceil(seconds));
      txt = `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
    }
    if (this._last.time !== txt) {
      this._last.time = txt;
      this.timeEl.textContent = txt;
    }
    const label = !Number.isFinite(seconds) ? 'NO LIMIT' : totalRounds ? `ROUND ${round}/${totalRounds}` : `ROUND ${round}`;
    if (this._last.round !== label) {
      this._last.round = label;
      this.roundEl.textContent = label;
    }
  }

  setLevel(name) {
    this.lvlEl.textContent = `AI · ${name.toUpperCase()}`;
  }

  /** Floating text at a screen position (0..1 coordinates). */
  callout(text, x, y, cls = 'hit', sub = '') {
    const el = document.createElement('div');
    el.className = `callout ${cls}`;
    el.innerHTML = sub ? `${text}<small>${sub}</small>` : text;
    el.style.left = `${Math.min(88, Math.max(12, x * 100))}%`;
    el.style.top = `${Math.min(80, Math.max(18, y * 100))}%`;
    this.callouts.appendChild(el);
    setTimeout(() => el.remove(), 950);
    while (this.callouts.children.length > 6) this.callouts.firstChild.remove();
  }

  combo(n) {
    if (n >= 2) {
      this.comboEl.textContent = `${n} HIT COMBO`;
      this.comboEl.classList.add('on');
    } else this.comboEl.classList.remove('on');
  }

  hint(text, ms = 2200) {
    if (this._last.hint === text && this.hintEl.classList.contains('on')) return;
    this._last.hint = text;
    this.hintEl.textContent = text;
    this.hintEl.classList.add('on');
    clearTimeout(this._hintTimer);
    if (ms) this._hintTimer = setTimeout(() => this.hintEl.classList.remove('on'), ms);
  }

  clearHint() {
    this.hintEl.classList.remove('on');
  }

  flashBanner(text, cls = '', hold = false) {
    const b = this.banner;
    b.className = '';
    void b.offsetWidth; // restart animation
    b.textContent = text;
    b.className = `${hold ? 'hold' : 'show'} ${cls}`;
  }

  hideBanner() {
    this.banner.className = '';
  }

  hurt() {
    this.damage.classList.add('on');
    requestAnimationFrame(() => requestAnimationFrame(() => this.damage.classList.remove('on')));
  }

  setStats(text) {
    if (this.stats.classList.contains('hidden')) return;
    if (this._last.stats !== text) {
      this._last.stats = text;
      this.stats.textContent = text;
    }
  }

  /** Mirror-view camera preview with the tracked skeleton. */
  drawPip(video, sample) {
    if (this.pip.classList.contains('hidden') || !video) return;
    const c = this.pipCanvas;
    const w = 240;
    const h = Math.round((w * video.videoHeight) / (video.videoWidth || 1)) || 180;
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    const g = this.pipCtx;
    g.save();
    g.translate(w, 0);
    g.scale(-1, 1);
    g.drawImage(video, 0, 0, w, h);
    g.restore();
    if (!sample) return;
    const n = sample.norm;
    const vis = sample.vis;
    g.lineWidth = 3;
    g.lineCap = 'round';
    for (const [a, b] of SKELETON_EDGES) {
      if (vis[a] < 0.4 || vis[b] < 0.4) continue;
      const left = a % 2 === 1 && b % 2 === 1;
      g.strokeStyle = left ? 'rgba(95,160,255,0.95)' : 'rgba(255,90,74,0.95)';
      g.beginPath();
      g.moveTo((1 - n[a * 2]) * w, n[a * 2 + 1] * h);
      g.lineTo((1 - n[b * 2]) * w, n[b * 2 + 1] * h);
      g.stroke();
    }
    g.fillStyle = '#fff';
    for (const i of [LM.L_WRIST, LM.R_WRIST, LM.NOSE, LM.L_ELBOW, LM.R_ELBOW, LM.L_SHOULDER, LM.R_SHOULDER]) {
      if (vis[i] < 0.4) continue;
      g.beginPath();
      g.arc((1 - n[i * 2]) * w, n[i * 2 + 1] * h, 3.5, 0, Math.PI * 2);
      g.fill();
    }
  }
}
