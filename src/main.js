// Shadow Boxing — game orchestrator.
//
//  Camera ─► PoseService (worker, back-pressured) ─► PoseFilter (One Euro + prediction)
//        ─► CameraDriver (landmarks → BodyPose) ─► Fighter.solve (rig + IK) ─► render
//  AI brain ─► ProceduralBoxer (IK punches) ─► Fighter.solve
//  Combat (swept-sphere hits) ─► FX / audio / HUD
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { Renderer } from './core/Renderer.js';
import { TIERS, detectTier, safeGet, safeSet } from './core/Quality.js';
import { Arena, blobShadow } from './scene/Arena.js';
import { Fighter, createBodyPose } from './avatar/Fighter.js';
import { ProceduralBoxer } from './avatar/ProceduralBoxer.js';
import { CameraDriver } from './avatar/CameraDriver.js';
import { ViewRig, VIEWS } from './core/Views.js';
import { OpponentAI } from './game/OpponentAI.js';
import { Combat } from './game/Combat.js';
import { Effects } from './fx/Effects.js';
import { Sfx } from './audio/Sfx.js';
import { Hud } from './ui/Hud.js';
import { PoseFilter } from './pose/PoseFilter.js';
import { LM } from './pose/landmarks.js';

const $ = (s) => document.querySelector(s);
const BASE = import.meta.env.BASE_URL;
const DISTANCE = 0.88; // starting root-to-root distance between the fighters (m)
const MIN_GAP = 0.62; // fighters can't walk through each other
const RING_LIMIT = 2.55; // |z| limit inside the ropes
// Opponent health (the player always has 100). 'endless' never ends in a KO:
// a knockdown refills the fighter and the fight goes on.
const HEALTH_PRESETS = { normal: 100, tough: 300, iron: 1000, endless: 300 };
const MATCH_PRESETS = {
  standard: { rounds: 3, seconds: 90 },
  long: { rounds: 5, seconds: 180 },
  unlimited: { rounds: 1, seconds: Infinity },
};
const PLAYER_GLOVE = 0x1d4ed8;
const AI_GLOVE = 0xc8102e;

// move → [callout label, power multiplier]
const PUNCH_INFO = {
  jab: ['JAB', 0.75],
  cross: ['CROSS', 1.0],
  hook: ['LEAD HOOK', 1.15],
  rhook: ['REAR HOOK', 1.2],
  uppercut: ['REAR UPPERCUT', 1.3],
  lupper: ['LEAD UPPERCUT', 1.15],
  overhand: ['OVERHAND', 1.35],
  body: ['BODY SHOT', 0.9],
  bodyhook: ['LIVER SHOT', 1.15],
  rbodyhook: ['BODY HOOK', 1.1],
  lelbow: ['LEAD ELBOW', 1.35],
  relbow: ['REAR ELBOW', 1.45],
};
// camera-detected punch kind + hand → move
const CAMERA_MOVES = {
  straight: ['jab', 'cross'],
  hook: ['hook', 'rhook'],
  upper: ['lupper', 'uppercut'],
  body: ['body', 'body'],
  bodyhook: ['bodyhook', 'rbodyhook'],
  overhand: ['overhand', 'overhand'],
  elbow: ['lelbow', 'relbow'],
};

const _v = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _camPos = new THREE.Vector3();
const _camLook = new THREE.Vector3();

class Game {
  constructor() {
    this.tier = detectTier();
    this.preset = TIERS[this.tier];
    this.mode = 'attract'; // attract | camera | keys
    this.phase = 'menu';
    this.difficulty = safeGet('sb.diff') || 'normal';
    this.stance = safeGet('sb.stance') || 'orthodox';
    this.tracker = safeGet('sb.tracker') || 'mediapipe'; // mediapipe = true 3D; movenet = 2D, fastest
    this.trackMode = safeGet('sb.track') || 'full'; // full body | arms only
    this.timeScale = 1;
    this.healthMode = safeGet('sb.health') || 'tough';
    this.matchMode = safeGet('sb.match') || 'long';
    this.maxAI = HEALTH_PRESETS[this.healthMode] || 100;
    this.clock = { round: 1, time: MATCH_PRESETS[this.matchMode].seconds };
    this.hp = { you: 100, ai: 100 };
    this.stats = this._freshStats();
    this.keys = new Set();
    this.poseService = null;
    this.filter = new PoseFilter();
    this.lastSample = null;
    this.poseDirty = false;
    this.calibProgress = 0;
    this.combo = 0;
    this.lastPlayerHitAt = 0;
    this.statsOn = false;
    this.fps = 60;
    this.camBlend = 0; // 0 = attract orbit, 1 = fight cam
    this.zPlayer = DISTANCE / 2; // fighters travel along the world Z axis
    this.zOpp = -DISTANCE / 2;
    this.playerAnchorZ = DISTANCE / 2; // where the player stood at calibration
  }

  _freshStats() {
    return { thrown: 0, landed: 0, blocked: 0, taken: 0, dodges: 0, maxCombo: 0, damage: 0 };
  }

  async init() {
    const loading = $('#loading');
    const msg = $('#loading .msg');
    loading.classList.remove('hidden');
    $('#menu').classList.add('hidden');
    msg.textContent = 'Loading arena…';

    await Promise.race([document.fonts?.load('400 100px Anton'), new Promise((r) => setTimeout(r, 1500))]).catch(() => {});

    this.renderer = new Renderer($('#scene'), this.tier, this.preset);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.05, 80);
    this.camera.position.set(3, 2, 3);
    this.renderer.setup(this.scene, this.camera);
    this.views = new ViewRig(this.camera);
    this.view = this.views.set(safeGet('sb.view') || 'side');
    this.arena = new Arena(this.scene, this.renderer.renderer, this.preset);

    // Two distinct CC0 MakeHuman boxers (generated by tools/character/)
    const loader = new GLTFLoader();
    const progress = {};
    const onProgress = (key) => (e) => {
      if (!e.total) return;
      // Hosts that gzip the files (e.g. GitHub Pages) report the compressed
      // size as total while loaded counts decompressed bytes, so clamp.
      progress[key] = Math.min(1, e.loaded / e.total);
      const p = (Object.values(progress).reduce((a, b) => a + b, 0) / 2) * 100;
      msg.textContent = `Loading fighters… ${Math.min(100, Math.round(p))}%`;
    };
    const [playerGltf, oppGltf] = await Promise.all([
      loader.loadAsync(`${BASE}models/boxer_player.glb`, onProgress('p')),
      loader.loadAsync(`${BASE}models/boxer_opponent.glb`, onProgress('o')),
      this.arena.loadEnvironment(`${BASE}env/studio.hdr`).catch((e) => console.warn('env', e)),
    ]);
    this.models = { player: playerGltf.scene, opponent: oppGltf.scene };
    this._buildFighters();

    this.effects = new Effects(this.scene);
    this.sfx = new Sfx();
    this.hud = new Hud();
    this.combat = new Combat();

    // Warm up: compile all shaders before the first visible frame (no hitch)
    this._update(1 / 60);
    this.renderer.renderer.compile(this.scene, this.camera);

    this._bindUI();
    loading.classList.add('hidden');
    this._showMenu();
    this.last = performance.now();
    this.renderer.renderer.setAnimationLoop((t) => this._frame(t));
  }

  _buildFighters() {
    if (this.player) {
      this.scene.remove(this.player.group, this.opponent.group, this.shadowA, this.shadowB);
    }
    // kit colours are baked into each boxer model
    this.player = new Fighter(this.models.player, { gloveColor: PLAYER_GLOVE, stance: this.stance, name: 'player' });
    this.opponent = new Fighter(this.models.opponent, { gloveColor: AI_GLOVE, stance: 'orthodox', name: 'opponent' });
    if (!this.preset.physicalGloves) {
      for (const f of [this.player, this.opponent]) {
        for (const s of ['Left', 'Right']) {
          const m = f.gloves[s].children[0];
          const old = m.material;
          m.material = new THREE.MeshStandardMaterial({ color: old.color, roughness: 0.35 });
          f.gloves[s].userData.leatherMat = m.material;
        }
      }
    }
    this.player.group.position.set(0, 0, DISTANCE / 2);
    this.player.group.rotation.y = Math.PI;
    this.opponent.group.position.set(0, 0, -DISTANCE / 2);
    this.scene.add(this.player.group, this.opponent.group);
    this.shadowA = blobShadow();
    this.shadowB = blobShadow();
    this.scene.add(this.shadowA, this.shadowB);

    this.playerPose = createBodyPose();
    this.oppPose = createBodyPose();
    this.playerBoxer = new ProceduralBoxer(this.player);
    this.oppBoxer = new ProceduralBoxer(this.opponent);
    this.oppAI = new OpponentAI(this.oppBoxer, this.opponent, this.player, this.difficulty);
    this.playerAI = new OpponentAI(this.playerBoxer, this.player, this.opponent, 'normal');
    this.cameraDriver = new CameraDriver(this.player, this.opponent);
    this._landed = { Left: -1, Right: -1 };
    this._thrownId = { Left: 0, Right: 0 };

    this.oppBoxer.onLaunch = (A) => {
      A.landed = false;
      this.opponent.breathe(A.m.power);
      this.opponent.effort(A.m.power);
      this.sfx?.whoosh(A.m.power);
    };
    this.oppBoxer.onImpact = (A) => this._onAIPunchEnd(A);
    this.playerBoxer.onLaunch = (A) => {
      this.player.breathe(A.m.power);
      if (!A.m.parry) this.player.effort(A.m.power);
      if (this.mode !== 'attract' && !A.m.parry) {
        this.stats.thrown++;
        this.sfx?.whoosh(A.m.power);
      }
    };
  }

  // ---------------------------------------------------------------- UI ----
  _bindUI() {
    $('#sel-diff').value = this.difficulty;
    $('#sel-health').value = this.healthMode;
    $('#sel-match').value = this.matchMode;
    $('#sel-health').onchange = (e) => {
      this.healthMode = e.target.value;
      safeSet('sb.health', this.healthMode);
    };
    $('#sel-match').onchange = (e) => {
      this.matchMode = e.target.value;
      safeSet('sb.match', this.matchMode);
    };
    $('#sel-quality').value = safeGet('sb.quality') || 'auto';
    $('#sel-stance').value = this.stance;
    $('#sel-tracker').value = this.tracker;
    $('#sel-track').value = this.trackMode;
    $('#sel-track').onchange = (e) => this._setTrackMode(e.target.value, true);
    $('#sel-tracker').onchange = (e) => {
      this.tracker = e.target.value;
      safeSet('sb.tracker', this.tracker);
    };
    $('#sel-diff').onchange = (e) => {
      this.difficulty = e.target.value;
      safeSet('sb.diff', this.difficulty);
    };
    $('#sel-quality').onchange = (e) => {
      safeSet('sb.quality', e.target.value);
      location.reload();
    };
    $('#sel-stance').onchange = (e) => {
      this.stance = e.target.value;
      safeSet('sb.stance', this.stance);
      this._buildFighters();
    };
    $('#btn-camera').onclick = () => this.startCamera();
    $('#btn-keys').onclick = () => this.startKeys();
    $('#btn-rematch').onclick = () => this._rematch();
    $('#btn-menu').onclick = () => this.quitToMenu();
    $('#btn-pause').onclick = () => this.quitToMenu();
    $('#btn-skip-calib').onclick = () => this.quitToMenu();
    $('#btn-stats').onclick = () => {
      this.statsOn = !this.statsOn;
      $('#stats').classList.toggle('hidden', !this.statsOn);
    };
    $('#btn-view').onclick = () => this._cycleView();
    $('#sel-view').value = this.view;
    $('#sel-view').onchange = (e) => this._setView(e.target.value);
    this._setView(this.view, true);
    $('#btn-mute').onclick = () => {
      this.sfx.setMuted(!this.sfx.muted);
      $('#btn-mute').textContent = this.sfx.muted ? '🔇' : '🔊';
    };

    addEventListener('keydown', (e) => {
      if (e.repeat) return;
      const k = e.key.toLowerCase();
      this.keys.add(k);
      if (k === 'escape' && this.phase !== 'menu') this.quitToMenu();
      if (k === '`' || k === 'f2') $('#btn-stats').click();
      if (k === 'v' && this.mode !== 'attract') this._cycleView();
      if (k === 't' && this.mode === 'camera') this._setTrackMode(this.trackMode === 'arms' ? 'full' : 'arms');
      if (this.mode === 'keys' && this.phase === 'fight') this._keyPunch(k);
    });
    addEventListener('keyup', (e) => this.keys.delete(e.key.toLowerCase()));
    addEventListener('blur', () => this.keys.clear());
  }

  _resetPositions() {
    this.zPlayer = this.playerAnchorZ = DISTANCE / 2;
    this.zOpp = -DISTANCE / 2;
    this.player.group.position.z = this.zPlayer;
    this.opponent.group.position.z = this.zOpp;
  }

  _setView(v, silent = false) {
    this.view = this.views.set(v);
    safeSet('sb.view', this.view);
    $('#sel-view').value = this.view;
    $('#btn-view').textContent = `🎥 ${this.views.label()}`;
    if (!silent && this.mode !== 'attract') this.hud.hint(`View: ${this.views.label()}  ·  press V to switch`, 1400);
  }

  _setTrackMode(m, silent = false) {
    this.trackMode = m === 'arms' ? 'arms' : 'full';
    safeSet('sb.track', this.trackMode);
    $('#sel-track').value = this.trackMode;
    if (!silent) this.hud.hint(this.trackMode === 'arms' ? 'Tracking: ARMS ONLY (body steady) · T to switch' : 'Tracking: FULL BODY · T to switch', 1800);
  }

  _cycleView() {
    const i = VIEWS.findIndex((x) => x.id === this.view);
    this._setView(VIEWS[(i + 1) % VIEWS.length].id);
  }

  _showMenu() {
    this._resetPositions();
    this.phase = 'menu';
    this.mode = 'attract';
    this.hp.you = 100;
    this.hp.ai = this.maxAI;
    for (const b of [this.playerBoxer, this.oppBoxer]) b.knockedOut = 0;
    this.oppAI.setDifficulty('normal');
    this.oppAI.enabled = this.playerAI.enabled = true;
    $('#menu').classList.remove('hidden');
    $('#result').classList.add('hidden');
    $('#calib').classList.add('hidden');
    $('#keys-help').classList.add('hidden');
    this.hud.show(false);
    this.hud.hideBanner();
  }

  quitToMenu() {
    const midFight = this.mode !== 'attract' && (this.phase === 'fight' || this.phase === 'break' || this.phase === 'countdown');
    if (midFight && (this.healthMode === 'endless' || this.matchMode === 'unlimited') && this.stats.thrown + this.stats.landed > 0) {
      this._showResult('session');
      return;
    }
    this.poseService?.stop();
    this.poseService = null;
    this.sfx.crowd(false);
    this.timeScale = 1;
    this._showMenu();
  }

  async startCamera() {
    this.sfx.unlock();
    if (!navigator.mediaDevices?.getUserMedia) {
      alert('Camera access is not available in this browser. Try the keyboard mode, or open the page over https / localhost.');
      return;
    }
    $('#menu').classList.add('hidden');
    const loading = $('#loading');
    const msg = $('#loading .msg');
    loading.classList.remove('hidden');
    try {
      const { PoseService } = await import('./pose/PoseService.js');
      this.filter = new PoseFilter();
      this.poseService = new PoseService({
        model: this.preset.poseModel,
        engine: this.tracker,
        onStatus: (s) => (msg.textContent = s || 'Ready'),
        // runtime recovery notices (camera reconnect, tracker restart)
        onNotice: (text) => (text ? this.hud.hint(text, 0) : this.hud.clearHint()),
        onResult: (r) => {
          this.filter.push(r);
          this.poseDirty = true;
        },
      });
      await this.poseService.start();
    } catch (err) {
      console.error(err);
      loading.classList.add('hidden');
      this.poseService?.stop();
      this.poseService = null;
      const denied = /denied|NotAllowed|Permission/i.test(String(err?.name) + String(err?.message));
      alert(denied ? 'Camera permission was denied. Allow camera access and try again, or use keyboard mode.' : `Could not start pose tracking: ${err?.message || err}`);
      this._showMenu();
      return;
    }
    loading.classList.add('hidden');
    this.mode = 'camera';
    this._prepareMatch();
    this.phase = 'calibrate';
    this.calibProgress = 0;
    $('#calib').classList.remove('hidden');
    $('#pip').classList.remove('hidden');
    this.hud.show(true);
  }

  startKeys() {
    this.sfx.unlock();
    $('#menu').classList.add('hidden');
    $('#pip').classList.add('hidden');
    $('#keys-help').classList.remove('hidden');
    this.mode = 'keys';
    this._prepareMatch();
    this.hud.show(true);
    this._countdown();
  }

  _prepareMatch() {
    this._resetPositions();
    this.maxAI = HEALTH_PRESETS[this.healthMode] || 100;
    this.match = MATCH_PRESETS[this.matchMode] || MATCH_PRESETS.long;
    this.hp.you = 100;
    this.hp.ai = this.maxAI;
    this.clock.round = 1;
    this.clock.time = this.match.seconds;
    this.stats = this._freshStats();
    this.combo = 0;
    for (const b of [this.playerBoxer, this.oppBoxer]) {
      b.knockedOut = 0;
      b.cancel();
    }
    this.oppAI.setDifficulty(this.difficulty);
    this.oppAI.enabled = false;
    this.playerAI.enabled = false;
    this.hud.setLevel(this.difficulty);
    this.hud.setHealth(100, 100);
    this.hud.setClock(1, this.match.seconds, this.match.rounds);
    this.sfx.crowd(true);
  }

  _countdown() {
    this.phase = 'countdown';
    this.oppAI.enabled = false;
    this.hud.flashBanner(Number.isFinite(this.match?.seconds ?? 0) ? `ROUND ${this.clock.round}` : 'NO LIMIT', 'gold');
    this.sfx.crowd(true);
    setTimeout(() => {
      if (this.phase !== 'countdown') return;
      this.hud.flashBanner('FIGHT!', 'red');
      this.sfx.bell(1);
      this.phase = 'fight';
      this.oppAI.enabled = true;
      this.oppAI.cooldown = 1.2;
    }, 1400);
  }

  _rematch() {
    $('#result').classList.add('hidden');
    $('#keys-help').classList.toggle('hidden', this.mode !== 'keys');
    this._prepareMatch();
    this.hud.show(true);
    this._countdown();
  }

  _keyPunch(k) {
    const map = { j: 'jab', k: 'cross', l: 'hook', o: 'rhook', i: 'uppercut', y: 'lupper', u: 'body', n: 'bodyhook', m: 'overhand', h: 'lelbow', b: 'relbow', p: 'parry' };
    const move = map[k];
    if (!move) return;
    const b = this.playerBoxer;
    if (b.queue.length > 1) return;
    if (move === 'parry') {
      // swat in front of your own face, across toward the incoming punch
      b.cancel();
      b.punch('parry', (out) => out.set(-0.02 * this.player.stanceSign, b.headY - 0.02, 0.42), { windup: 0.01 });
      return;
    }
    const body = move === 'body' || move === 'bodyhook';
    b.punch(move, (out) => {
      _v.copy(body ? this.opponent.bellyWorld : this.opponent.headWorld);
      if (body) _v.lerp(this.opponent.chestWorld, 0.35);
      this.player.toRootPos(_v, out);
      out.z += 0.1;
      return out;
    });
  }

  /** Which move landed (for labels & power). */
  _playerMove(side) {
    if (this.mode === 'camera') {
      const st = this.cameraDriver.punch[side];
      const lead = (side === 'Left') === (this.player.stanceSign > 0);
      return (CAMERA_MOVES[st.kind] || CAMERA_MOVES.straight)[lead ? 0 : 1];
    }
    return this.playerBoxer.action?.name || 'cross';
  }

  // ----------------------------------------------------------- frame ----
  _frame(t) {
    const rawDt = Math.min(0.05, (t - this.last) / 1000);
    this.last = t;
    this.fps += (1 / Math.max(rawDt, 1e-3) - this.fps) * 0.05;
    this.renderer.adapt(rawDt * 1000);

    // hit-stop / slow motion give punches weight
    let scale = this.timeScale;
    if (this.effects.hitStop > 0) {
      this.effects.hitStop -= rawDt;
      scale *= 0.08;
    }
    this._update(rawDt * scale, rawDt);
    this.renderer.render();
    this._hudTick(rawDt);
  }

  _update(dt, realDt = dt) {
    const now = performance.now();

    // ---- Player body source
    let usingCamera = false;
    if (this.mode === 'camera' && this.poseService && !this.filter.isLost(500) && this.filter.hasData) {
      const s = this.filter.sample(now);
      this.lastSample = s;
      const armsOnly = this.trackMode === 'arms';
      if (armsOnly) {
        // steady procedural stance for torso/hips/head/legs; camera drives the arms
        this.playerAI.enabled = false;
        this.playerBoxer.idleAmp = 0.6;
        this.playerBoxer.blockTarget = 0;
        this.playerBoxer.update(dt, this.playerPose);
      }
      this.cameraDriver.update(s, this.playerPose, this.poseService.aspect, realDt, armsOnly);
      usingCamera = true;
    }
    if (!usingCamera) {
      this.playerAI.enabled = this.mode === 'attract';
      if (this.playerAI.enabled) this.playerAI.update(dt);
      if (this.mode === 'keys') this._keyDefence();
      this.playerBoxer.update(dt, this.playerPose);
    }
    if (usingCamera && this.playerBoxer.knockedOut > 0) this.playerBoxer.update(dt, this.playerPose);

    // ---- Opponent
    this.oppAI.update(dt);
    this.oppBoxer.update(dt, this.oppPose);

    // ---- Footwork (who stands where along the fight axis)
    this._footwork(dt, realDt, usingCamera);

    // ---- Solve skeletons
    this.player.solve(this.playerPose, usingCamera ? realDt : dt);
    this.opponent.solve(this.oppPose, dt);

    // telegraph glow on the opponent's loaded glove (a readable "tell")
    const tele = this.oppBoxer.action && this.oppBoxer.action.phase === 'windup' ? this.oppBoxer.telegraph : 0;
    for (const s of ['Left', 'Right']) {
      const m = this.opponent.gloves[s].userData.leatherMat;
      const on = this.oppBoxer.action && this.oppBoxer.action.side === s ? tele : 0;
      m.emissive.setRGB(on * 0.9, on * 0.25, on * 0.1);
    }

    // ---- Combat
    if (this.phase === 'fight' || this.mode === 'attract') this._combat(usingCamera);

    // ---- World
    this._placeShadow(this.shadowA, this.player);
    this._placeShadow(this.shadowB, this.opponent);
    this.arena.update(dt);
    this.effects.update(realDt);
    this._updateCamera(realDt);
    this._updateMatch(realDt);
  }

  _keyDefence() {
    const b = this.playerBoxer;
    const K = this.keys;
    b.blockTarget = K.has(' ') ? 1 : 0;
    const slip = (K.has('q') ? 1 : 0) - (K.has('e') ? 1 : 0);
    b.slip += (slip * 0.22 - b.slip) * 0.3;
    b.duck += ((K.has('s') || K.has('arrowdown') ? 0.26 : 0) - b.duck) * 0.3;
  }

  /**
   * Movement along the fight axis. Camera: stepping toward the webcam walks the
   * fighter forward, stepping away retreats (CameraDriver.advance). The AI
   * manages distance like a boxer: it hovers just outside range, steps in to
   * attack, gives ground under pressure and gets pinned on the ropes.
   */
  _footwork(dt, realDt, usingCamera) {
    if (this.mode === 'attract' || this.phase === 'menu') {
      this.player.group.position.z = this.zPlayer;
      this.opponent.group.position.z = this.zOpp;
      return;
    }
    const ko = this.phase === 'ko' || this.phase === 'result';

    // --- player
    let target = this.zPlayer;
    if (usingCamera) {
      target = this.playerAnchorZ - this.cameraDriver.advance; // player faces -Z
    } else if (this.mode === 'keys' && !ko) {
      const K = this.keys;
      const dir = (K.has('d') || K.has('arrowright') ? 1 : 0) - (K.has('a') || K.has('arrowleft') ? 1 : 0);
      target = this.zPlayer - dir * 1.6 * dt;
    }
    if (!ko) {
      const maxStep = 2.8 * realDt;
      this.zPlayer += THREE.MathUtils.clamp(target - this.zPlayer, -maxStep, maxStep);
    }

    // --- opponent: preferred gap depends on what it's doing
    if (!ko) {
      const ai = this.oppAI;
      const b = this.oppBoxer;
      let gap = 0.9; // hover at the edge of range: the player has to commit to reach
      if (b.busy || ai.cooldown < 0.55) gap = 0.84; // step in to attack
      if (ai.stun > 0) gap = 1.15; // hurt: back off
      gap += Math.sin(performance.now() / 900) * 0.04; // restless feet
      const want = this.zPlayer - gap;
      const maxStep = (gap > 0.9 ? 1.1 : 1.6) * dt;
      this.zOpp += THREE.MathUtils.clamp(want - this.zOpp, -maxStep, maxStep);
      this.zOpp = Math.max(-RING_LIMIT, Math.min(this.zOpp, RING_LIMIT - MIN_GAP));
    }

    // --- constraints: ropes and bodies
    this.zPlayer = Math.min(RING_LIMIT, Math.max(this.zPlayer, this.zOpp + MIN_GAP));
    this.player.group.position.z = this.zPlayer;
    this.opponent.group.position.z = this.zOpp;
  }

  _combat(usingCamera) {
    const P = this.player;
    const O = this.opponent;

    // player → opponent
    // Camera: a glove is live while a detected punch is in flight and that
    // punch hasn't landed yet (one hit per punch; retract to re-arm).
    const punch = this.cameraDriver.punch;
    // returns false, 'glove' or 'elbow' (which part of the arm is striking)
    const playerLive = usingCamera
      ? (side) => punch[side].active && punch[side].id !== this._landed[side] && (punch[side].kind === 'elbow' ? 'elbow' : 'glove')
      : (side) => {
          const A = this.playerBoxer.action;
          const on = !!A && !A.m.parry && A.side === side && (A.phase === 'strike' || A.phase === 'hold');
          return on && (A.m.elbow ? 'elbow' : 'glove');
        };
    if (usingCamera) this._detectThrows();
    for (const e of this.combat.check(P, O, playerLive, this.oppBoxer.blockAmt > 0.5)) this._onPlayerHit(e);

    // opponent → player
    const oppLive = (side) => {
      const A = this.oppBoxer.action;
      const on = !!A && A.side === side && (A.phase === 'strike' || A.phase === 'hold');
      return on && (A.m.elbow ? 'elbow' : 'glove');
    };
    for (const e of this.combat.check(O, P, oppLive, this._playerGuarding(usingCamera))) this._onAIHit(e);
  }

  /**
   * Which of your gloves can stop a punch: a glove held up by your face
   * covers it ("face save"); a glove swatting fast is a parry.
   */
  _playerGuarding(usingCamera) {
    const P = this.player;
    if (!usingCamera) {
      const b = this.playerBoxer;
      return (side) => b.blockAmt > 0.5 || (b.action?.m.parry && b.action.side === side);
    }
    return (side) => P.gloveWorld[side].distanceTo(P.headWorld) < 0.32 || P.gloveVel[side].length() > 1.8;
  }

  /** Counts thrown punches + whoosh sound for camera play. */
  _detectThrows() {
    const punch = this.cameraDriver.punch;
    for (const s of ['Left', 'Right']) {
      const st = punch[s];
      if (st.active && st.id !== this._thrownId[s]) {
        this._thrownId[s] = st.id;
        this.player.breathe(1);
        this.player.effort(1);
        if (this.phase === 'fight') {
          this.stats.thrown++;
          this.sfx.whoosh(Math.min(1.4, 0.6 + st.rate * 0.15));
        }
      }
    }
  }

  _screen(p) {
    _v2.copy(p).project(this.camera);
    return [(_v2.x + 1) / 2, (1 - _v2.y) / 2];
  }

  _onPlayerHit(e) {
    if (this.mode === 'camera') this._landed[e.side] = this.cameraDriver.punch[e.side].id;
    const attract = this.mode === 'attract';
    const O = this.opponent;
    if (e.zone === 'block') {
      this.effects.impact(e.point, e.dir, 0.4, true);
      this.sfx.block();
      if (!attract) {
        this.stats.blocked++;
        const [x, y] = this._screen(e.point);
        this.hud.callout('BLOCKED', x, y, 'block');
      }
      return;
    }
    // elbows travel a short arc (lower speed) but land with the whole body behind them
    const speedK = attract ? 1 : THREE.MathUtils.clamp(e.speed / 4.5, e.striker === 'elbow' ? 1.05 : 0.75, 1.6);
    const move = attract ? 'cross' : this._playerMove(e.side);
    const [moveLabel, mult] = PUNCH_INFO[move] || PUNCH_INFO.cross;
    const power = speedK * mult;
    const counter = this.oppAI.onHit(power) || performance.now() < (this.counterUntil || 0);
    this.counterUntil = 0;
    O.hit(O.toRootDir(e.dir, _v).clone(), power * (e.zone === 'head' ? 1 : 0.8), e.zone);
    this.effects.impact(e.point, e.dir, power, false);
    this.sfx.punch(power);
    if (power > 1.1) {
      this.sfx.roar(power - 0.6);
      this.arena.cheer(power);
    }
    if (attract) return;
    if (power > 1.2) this.views.focusHit(); // cinematic: cut to the reaction

    let dmg = (e.zone === 'head' ? 7 : 5) * power * (counter ? 1.4 : 1);
    dmg = Math.round(dmg);
    this.hp.ai = Math.max(0, this.hp.ai - dmg);
    this.stats.landed++;
    this.stats.damage += dmg;
    const now = performance.now();
    this.combo = now - this.lastPlayerHitAt < 1500 ? this.combo + 1 : 1;
    this.lastPlayerHitAt = now;
    this.stats.maxCombo = Math.max(this.stats.maxCombo, this.combo);
    const [x, y] = this._screen(e.point);
    const label = counter ? `COUNTER ${moveLabel}` : moveLabel;
    this.hud.callout(label, x, y - 0.05, counter || power > 1.3 ? 'crit' : 'hit', `-${dmg}`);
    this.hud.combo(this.combo);
    if (this.hp.ai <= 0) this.healthMode === 'endless' ? this._knockdown('ai') : this._knockout('ai');
  }

  _onAIHit(e) {
    const attract = this.mode === 'attract';
    const P = this.player;
    const A = this.oppBoxer.action;
    if (A) A.landed = true;
    if (e.zone === 'block') {
      this.effects.impact(e.point, e.dir, 0.4, true);
      this.sfx.block();
      if (attract) return;
      const [x, y] = this._screen(e.point);
      const parry =
        this.mode === 'camera'
          ? P.gloveVel[e.defSide].length() > 1.8
          : !!this.playerBoxer.action?.m.parry;
      if (parry) {
        // knocked off line: the AI is off balance and open for a counter
        this.oppBoxer.cancel();
        this.oppAI.stun = Math.max(this.oppAI.stun, 0.6);
        this.counterUntil = performance.now() + 900;
        this.stats.parries = (this.stats.parries || 0) + 1;
        this.hud.callout('PARRY!', x, y, 'crit');
      } else {
        this.hp.you = Math.max(0, this.hp.you - 1); // blocked shots still sting a little
        this.stats.blocksMade = (this.stats.blocksMade || 0) + 1;
        this.hud.callout('BLOCKED', x, y, 'block');
      }
      return;
    }
    const power = A ? A.m.power : 1;
    P.hit(P.toRootDir(e.dir, _v).clone(), power * (e.zone === 'head' ? 1.1 : 0.8), e.zone);
    this.effects.impact(e.point, e.dir, power, false);
    this.sfx.punch(power * 0.9);
    this.playerAI.onHit?.(power);
    if (attract) return;
    const dmg = Math.round((e.zone === 'head' ? 6 : 4.5) * power * this.oppAI.diff.dmg);
    this.hp.you = Math.max(0, this.hp.you - dmg);
    this.stats.taken += dmg;
    this.combo = 0;
    this.hud.combo(0);
    this.hud.hurt();
    this.effects.shake = Math.max(this.effects.shake, 0.6);
    if (this.hp.you <= 0) this.healthMode === 'endless' ? this._knockdown('you') : this._knockout('you');
  }

  _onAIPunchEnd(A) {
    if (this.mode === 'attract' || this.phase !== 'fight' || A.landed) return;
    // the punch reached its end without touching the player: a clean dodge
    _v.copy(A.target);
    this.opponent.group.localToWorld(_v);
    if (_v.distanceTo(this.player.headWorld) > 0.16) {
      this.stats.dodges++;
      // name the defence by how the head got out of the way
      const h = this.player.headWorld;
      const dy = h.y - _v.y;
      const dz = h.z - _v.z; // + = further from the opponent
      const label = dy < -0.12 ? 'DUCKED!' : dz > 0.12 ? 'PULLED BACK!' : 'SLIPPED!';
      const [x, y] = this._screen(h);
      this.hud.callout(label, x, y - 0.1, 'crit');
    }
  }

  /** Game face during the fight, relaxed between rounds, KO / winner smile. */
  _faceState() {
    const fighting = this.phase === 'fight' || this.phase === 'countdown' || this.mode === 'attract';
    const focus = fighting ? 0.75 : this.phase === 'break' ? 0.2 : 0.35;
    for (const [f, b] of [[this.player, this.playerBoxer], [this.opponent, this.oppBoxer]]) {
      f.focusLevel = focus;
      f.koLevel = b.knockedOut;
    }
    const won = this.phase === 'ko' || this.phase === 'result';
    this.player.smileLevel = won && this.koLoser === 'ai' ? 1 : 0;
    this.opponent.smileLevel = won && this.koLoser === 'you' ? 1 : 0;
  }

  /** Endless mode: the fighter goes down, gets back up at full health, fight on. */
  _knockdown(who) {
    if (who === 'ai') {
      this.stats.knockdowns = (this.stats.knockdowns || 0) + 1;
      this.hp.ai = this.maxAI;
      this.oppAI.stun = 2.2; // wobbly: backs off before re-engaging
      this.oppBoxer.cancel();
      this.hud.flashBanner(`KNOCKDOWN ×${this.stats.knockdowns}`, 'gold');
    } else {
      this.stats.timesDown = (this.stats.timesDown || 0) + 1;
      this.hp.you = 100;
      this.oppAI.cooldown = Math.max(this.oppAI.cooldown, 2.5);
      this.hud.flashBanner('GET UP!', 'red');
    }
    this.sfx.roar(1.3);
    this.sfx.bell(1);
    this.arena.cheer(2.5);
  }

  _knockout(loser) {
    this.phase = 'ko';
    this.oppAI.enabled = false;
    this.oppBoxer.cancel();
    const boxer = loser === 'ai' ? this.oppBoxer : this.playerBoxer;
    this.koLoser = loser;
    this.koT = 0;
    this.timeScale = 0.35;
    this.hud.flashBanner('K.O.!', loser === 'ai' ? 'gold' : 'red', true);
    this.sfx.roar(1.5);
    this.sfx.bell(3);
    this.arena.cheer(3);
    boxer.cancel();
    this._koBoxer = boxer;
  }

  _updateMatch(dt) {
    this._faceState();
    if (this.phase === 'ko') {
      this.koT += dt;
      if (this._koBoxer) this._koBoxer.knockedOut = Math.min(1, this._koBoxer.knockedOut + dt * 1.2);
      if (this.koT > 1.4) this.timeScale = 1;
      if (this.koT > 3.6) this._showResult(this.koLoser === 'ai' ? 'win-ko' : 'lose-ko');
      return;
    }
    if (this.mode === 'attract') {
      // keep the demo fight going forever
      return;
    }
    if (this.phase === 'fight') {
      this.clock.time -= dt;
      if (this.clock.time <= 0) {
        this.sfx.bell(2);
        this.oppAI.enabled = false;
        this.oppBoxer.cancel();
        if (this.clock.round >= this.match.rounds) {
          const you = this.hp.you / 100;
          const ai = this.hp.ai / this.maxAI;
          this._showResult(you > ai ? 'win-dec' : you < ai ? 'lose-dec' : 'draw');
        } else {
          this.phase = 'break';
          this.breakT = 6;
          this.hud.flashBanner('END OF ROUND', 'gold', true);
        }
      }
    } else if (this.phase === 'break') {
      this.breakT -= dt;
      if (this.breakT <= 0) {
        this.clock.round++;
        this.clock.time = this.match.seconds;
        this.hp.you = Math.min(100, this.hp.you + 12);
        this.hp.ai = Math.min(this.maxAI, this.hp.ai + this.maxAI * 0.12);
        this.hud.hideBanner();
        this._countdown();
      }
    } else if (this.phase === 'calibrate') {
      this._calibrate(dt);
    }
    const aiPct = (this.hp.ai / this.maxAI) * 100;
    this.hud.setHealth(this.hp.you, aiPct);
    // hurt fighters breathe heavily through the mouth
    this.player.fatigue = Math.max(0, 1 - this.hp.you / 45);
    this.opponent.fatigue = Math.max(0, 1 - aiPct / 45);
    this.hud.setClock(this.clock.round, this.clock.time, this.match?.rounds);
    if (this.combo && performance.now() - this.lastPlayerHitAt > 1500) {
      this.combo = 0;
      this.hud.combo(0);
    }
    // coaching hints
    if (this.mode === 'camera' && this.phase === 'fight') {
      if (this.filter.isLost(600)) this.hud.hint('Step back into view 👀', 1200);
      else if (this.lastSample && Math.min(this.lastSample.vis[LM.L_SHOULDER], this.lastSample.vis[LM.R_SHOULDER]) < 0.5)
        this.hud.hint('Move back so your shoulders are visible', 1200);
    }
  }

  _calibrate(dt) {
    const s = this.lastSample;
    let body = false;
    let guard = false;
    if (s && !this.filter.isLost(300)) {
      const v = s.vis;
      const n = s.norm;
      body = v[LM.L_SHOULDER] > 0.6 && v[LM.R_SHOULDER] > 0.6 && v[LM.NOSE] > 0.5;
      const shY = (n[LM.L_SHOULDER * 2 + 1] + n[LM.R_SHOULDER * 2 + 1]) / 2;
      const shX = (n[LM.L_SHOULDER * 2] + n[LM.R_SHOULDER * 2]) / 2;
      const shW = Math.abs(n[LM.L_SHOULDER * 2] - n[LM.R_SHOULDER * 2]);
      // wrists up near the chin and tucked in front of the body
      const tucked = (i) => Math.abs(n[i * 2] - shX) < shW * 0.9 && n[i * 2 + 1] < shY + shW * 0.15;
      guard = body && v[LM.L_WRIST] > 0.4 && v[LM.R_WRIST] > 0.4 && tucked(LM.L_WRIST) && tucked(LM.R_WRIST);
    }
    const checks = document.querySelectorAll('.checks span');
    checks[0].classList.toggle('ok', body);
    checks[1].classList.toggle('ok', guard);
    $('.calib-msg').textContent = !body
      ? 'Step back until your head and chest are in view.'
      : !guard
        ? 'Now raise both gloves up to your chin — like a boxer.'
        : 'Hold it…';
    this.calibProgress = Math.max(0, Math.min(1, this.calibProgress + (body && guard ? dt / 1.1 : -dt / 2)));
    $('.ring .prog').style.strokeDashoffset = String(327 * (1 - this.calibProgress));
    if (this.calibProgress >= 1) {
      this.cameraDriver.calibrate(s, this.poseService.aspect);
      this.playerAnchorZ = this.zPlayer;
      $('#calib').classList.add('hidden');
      this._countdown();
    }
  }

  _showResult(kind) {
    this.phase = 'result';
    this.oppAI.enabled = false;
    this.timeScale = 1;
    const titles = {
      'win-ko': ['YOU WIN', 'KNOCKOUT VICTORY'],
      'win-dec': ['YOU WIN', 'ON POINTS'],
      'lose-ko': ['KNOCKED OUT', 'GET BACK UP, CHAMP'],
      'lose-dec': ['YOU LOSE', 'ON POINTS'],
      draw: ['DRAW', 'SPLIT DECISION'],
      session: ['SESSION OVER', this.healthMode === 'endless' ? 'ENDLESS FIGHT' : 'NO TIME LIMIT'],
    };
    const [title, kick] = titles[kind];
    $('.result-title').textContent = title;
    $('.result-kicker').textContent = kick;
    const st = this.stats;
    const acc = st.thrown ? Math.round((st.landed / Math.max(st.thrown, st.landed)) * 100) : 0;
    $('.result-grid').innerHTML = [
      [st.landed, 'Punches landed'],
      [`${acc}%`, 'Accuracy'],
      [st.maxCombo, 'Best combo'],
      [st.damage, 'Damage dealt'],
      [st.dodges, 'Dodges'],
      [st.taken, 'Damage taken'],
      ...(this.healthMode === 'endless' ? [[st.knockdowns || 0, 'Knockdowns scored'], [st.timesDown || 0, 'Times you went down']] : []),
    ]
      .map(([v, l]) => `<div><b>${v}</b><span>${l}</span></div>`)
      .join('');
    setTimeout(() => {
      this.hud.hideBanner();
      this.hud.show(false);
      $('#keys-help').classList.add('hidden');
      $('#result').classList.remove('hidden');
    }, 400);
  }

  _placeShadow(shadow, f) {
    f.rig.bones.Hips.getWorldPosition(_v);
    shadow.position.set(_v.x, 0.004, _v.z);
    shadow.scale.setScalar(1 + (1.05 - _v.y) * 0.3);
  }

  _updateCamera(dt) {
    const fight = this.mode !== 'attract';
    this.camBlend += ((fight ? 1 : 0) - this.camBlend) * (1 - Math.exp(-dt * 2.2));
    const k = this.camBlend;
    const t = performance.now() / 1000;

    // attract: slow orbit; on wide screens the fighters sit right of the menu
    const a = t * 0.12 + 0.6;
    const orbit = _v.set(Math.sin(a) * 3.6, 1.7 + Math.sin(t * 0.3) * 0.15, Math.cos(a) * 3.6);
    const orbitLook = _v2.set(0, 1.15, 0);
    if (this.camera.aspect < 0.8) orbitLook.y -= 1.15; // portrait: fighters above the menu
    if (this.camera.aspect > 1.1) {
      const side = Math.min(1.1, (this.camera.aspect - 1) * 1.2);
      orbitLook.x -= Math.cos(a) * side;
      orbitLook.z += Math.sin(a) * side;
    }

    // fight: the selected view (side / TPP / FPP / front / top / cinematic)
    this.views.update({ player: this.player, opponent: this.opponent, zPlayer: this.zPlayer, zOpp: this.zOpp, aspect: this.camera.aspect, dt });
    orbit.lerp(this.views.pos, k);
    orbitLook.lerp(this.views.look, k);
    const orbitFov = this.camera.aspect < 1 ? 70 : 45;
    const fov = orbitFov + (this.views.fov - orbitFov) * k;
    const near = k > 0.5 ? this.views.near : 0.05;
    if (Math.abs(this.camera.fov - fov) > 0.01 || this.camera.near !== near) {
      this.camera.fov = fov;
      this.camera.near = near;
      this.camera.updateProjectionMatrix();
    }
    // first person: hide your own head so the camera isn't inside it
    this.player.setHeadVisible(!(fight && this.views.view === 'fpp' && k > 0.5));

    this.camera.position.copy(orbit).add(this.effects.shakeOffset);
    this.camera.lookAt(orbitLook);
  }

  _hudTick(dt) {
    // camera preview: redrawn from the live video at ~20 Hz on its own, so it
    // never looks frozen even if a tracking result is late
    this._pipT = (this._pipT || 0) + dt;
    if (this.mode === 'camera' && (this.poseDirty || this._pipT > 0.05)) {
      this.poseDirty = false;
      this._pipT = 0;
      this.hud.drawPip(this.poseService?.video, this.lastSample);
    }
    this._statT = (this._statT || 0) + dt;
    if (this.statsOn && this._statT > 0.25) {
      this._statT = 0;
      const r = this.renderer;
      const ps = this.poseService?.stats;
      let s = `FPS      ${this.fps.toFixed(0)}\nTier     ${this.tier}\nScale    ${r.dpr.toFixed(2)}x`;
      if (ps) {
        s += `\nBody     ${this.trackMode === 'arms' ? 'arms only' : 'full body'}\nTracker  ${this.tracker === 'movenet' ? 'MoveNet 2D→3D' : 'MediaPipe 3D (' + (ps.model || this.preset.poseModel) + ')'}\nPose     ${ps.poseHz.toFixed(0)} Hz (${ps.delegate}/${ps.mode})\nInfer    ${ps.inferMs.toFixed(1)} ms\nCam→pose ${ps.latencyMs.toFixed(0)} ms\nRecover  ${ps.stalls} stalls · ${ps.restarts} restarts · ${ps.reconnects} cam`;
      }
      this.hud.setStats(s);
    }
  }
}

const game = new Game();
window.__game = game;
game.init().catch((err) => {
  console.error(err);
  $('#loading .msg').textContent = `Failed to start: ${err.message}`;
});
