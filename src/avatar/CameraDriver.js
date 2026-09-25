// Converts filtered MediaPipe landmarks into a BodyPose for the player's
// avatar. MediaPipe world landmarks are metric, hip-centred, x = image right,
// y = down, z = away from camera. Our root space is x = character LEFT,
// y = up, z = forward. A person facing the camera has their left side on the
// image right and "forward" toward the camera, so the mapping is (x, -y, -z).
import * as THREE from 'three';
import { LM } from '../pose/landmarks.js';

const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c = new THREE.Vector3();
const _vert = new THREE.Vector3();

function wp(world, i, out) {
  const o = i * 3;
  return out.set(world[o], -world[o + 1], -world[o + 2]);
}

const smooth01 = (e0, e1, x) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

// Punch detection tuning — deliberately demanding so only real punches count.
// e = arm extension (0 = fist at shoulder, 1 = arm straight).
const PUNCH = {
  onsetRate: 2.2, // straight: extension must grow at least this fast (1/s)
  onsetFwd: 1.2, // straight: fist moving forward (m/s, chest frame)
  onsetSpeed: 1.8, // straight: overall fist speed — a lazy push never gets here
  confirmFwd: 0.23, // straight/body: fist must travel this far forward (m)
  confirmExt: 0.8, // straight: arm must reach this extension…
  confirmGain: 0.28, // …and extend at least this much beyond the guard
  confirmAim: 0.5, // fist must end up pointing forward from the shoulder
  whipSpeed: 2.3, // hook/uppercut: fist speed relative to the chest (m/s)
  whipDir: 1.6, // …moving across the body or upward (m/s)
  confirmPath: 0.22, // hook/uppercut: fist must travel this far (m)
  candidateTime: 0.25, // an unconfirmed candidate is dropped after this (s)
  cooldown: 0.18, // per-hand rest after a punch (s)
};
const DEPTH_GAIN = 1.2; // mild correction for MediaPipe's compressed arm depth

const newPunchState = () => ({
  active: false,
  id: 0,
  e: 0,
  eRest: 0.45,
  prevE: 0,
  rate: 0,
  kind: 'straight',
  t: 0,
  travelW: 0,
  wOut: 0,
  path: 0,
  phase: 'idle', // idle → candidate → active → idle (+cooldown)
  hist: [], // recent fist positions (chest frame) so a punch's start isn't lost
  guardRel: new THREE.Vector3(0, 0.05, 0.24), // fist position in guard (calibrated)
  cool: 0,
  maxE: 0,
  maxY: 0,
  sumV: new THREE.Vector3(),
  originE: new THREE.Vector3(),
  pathE: 0,
  frozen: false,
  origin: new THREE.Vector3(),
  lostFor: 0,
});

/**
 * Apparent body size in the image (units of image height). Shoulder width is
 * corrected for torso yaw (so a punch's shoulder turn isn't read as stepping
 * back); torso height is added when hips are visible (yaw-independent).
 */
function bodySize(sample, aspect) {
  const n = sample.norm;
  const v = sample.vis;
  const dx = (n[LM.L_SHOULDER * 2] - n[LM.R_SHOULDER * 2]) * aspect;
  const dy = n[LM.L_SHOULDER * 2 + 1] - n[LM.R_SHOULDER * 2 + 1];
  let sw = Math.hypot(dx, dy);
  if (sample.world && !sample.lifted) {
    const w = sample.world;
    const wx = w[LM.L_SHOULDER * 3] - w[LM.R_SHOULDER * 3];
    const wz = w[LM.L_SHOULDER * 3 + 2] - w[LM.R_SHOULDER * 3 + 2];
    const cos = Math.abs(wx) / (Math.hypot(wx, wz) || 1);
    sw /= Math.max(0.6, cos);
  }
  const hipsVis = Math.min(v[LM.L_HIP], v[LM.R_HIP]) > 0.5;
  let th = 0;
  if (hipsVis) {
    const sx = (n[LM.L_SHOULDER * 2] + n[LM.R_SHOULDER * 2]) / 2;
    const sy = (n[LM.L_SHOULDER * 2 + 1] + n[LM.R_SHOULDER * 2 + 1]) / 2;
    const hx = (n[LM.L_HIP * 2] + n[LM.R_HIP * 2]) / 2;
    const hy = (n[LM.L_HIP * 2 + 1] + n[LM.R_HIP * 2 + 1]) / 2;
    th = Math.hypot((sx - hx) * aspect, sy - hy);
  }
  // lifted (2D) data: shoulder width is confounded by torso turn, so rely on
  // torso height alone whenever it's visible
  if (sample.lifted && th > 0) sw = 0;
  return { sw, th };
}

// Assumed distance from the camera at calibration (m). Only scales how far
// the fighter walks per real step; ~2 m is typical for webcam boxing.
const CALIB_DISTANCE = 2.0;

export class CameraDriver {
  constructor(fighter, opponent) {
    this.fighter = fighter;
    this.opponent = opponent;
    // per-arm punch state (read by the game for hit gating / stats)
    this.punch = {
      Left: newPunchState(),
      Right: newPunchState(),
    };
    this._wristPrev = { Left: new THREE.Vector3(), Right: new THREE.Vector3() };
    this._wristVel = { Left: new THREE.Vector3(), Right: new THREE.Vector3() };
    this._relE = { Left: new THREE.Vector3(), Right: new THREE.Vector3() };
    this._elbowPrevRel = { Left: new THREE.Vector3(), Right: new THREE.Vector3() };
    this._elbowVel = { Left: new THREE.Vector3(), Right: new THREE.Vector3() };
    this._uRight = new THREE.Vector3();
    this._uUp = new THREE.Vector3();
    this._ax = new THREE.Vector3();
    this._ay = new THREE.Vector3();
    this._az = new THREE.Vector3();
    this.armsOnly = false;
    this._bx = new THREE.Vector3();
    this._by = new THREE.Vector3();
    this._bz = new THREE.Vector3();
    this.calib = null;
    this.guardPose = {
      Left: { upper: new THREE.Vector3(0.25, -0.75, 0.45).normalize(), fore: new THREE.Vector3(-0.25, 0.75, 0.55).normalize() },
      Right: { upper: new THREE.Vector3(-0.25, -0.75, 0.45).normalize(), fore: new THREE.Vector3(0.25, 0.75, 0.55).normalize() },
    };
    this.pts = {};
    for (const k of ['ls', 'rs', 'le', 're', 'lw', 'rw', 'lh', 'rh', 'nose', 'lear', 'rear']) this.pts[k] = new THREE.Vector3();
    this._headUp = new THREE.Vector3();
    this._headRight = new THREE.Vector3();
    this.head = { up: this._headUp, right: this._headRight };
    this.hipsTarget = new THREE.Vector3();
    this.advance = 0; // metres the fighter has walked forward (from camera distance)
    this._ratio = 1;
    this.present = 0; // 0..1 confidence the user is tracked
  }

  /** Neutral reference captured while the user holds their guard. */
  calibrate(sample, aspect) {
    const n = sample.norm;
    const sx = (n[LM.L_SHOULDER * 2] + n[LM.R_SHOULDER * 2]) / 2;
    const sy = (n[LM.L_SHOULDER * 2 + 1] + n[LM.R_SHOULDER * 2 + 1]) / 2;
    const sw = Math.hypot((n[LM.L_SHOULDER * 2] - n[LM.R_SHOULDER * 2]) * aspect, n[LM.L_SHOULDER * 2 + 1] - n[LM.R_SHOULDER * 2 + 1]);
    const size = bodySize(sample, aspect);
    this.calib = { sx, sy, sw: Math.max(sw, 0.05), size };
    this.advance = 0;
    this._ratio = 1;
    // the user's guard extension becomes each arm's "resting" reference
    for (const s of ['Left', 'Right']) {
      this.punch[s].eRest = THREE.MathUtils.clamp(this.punch[s].e, 0.3, 0.6);
      this.punch[s].guardRel.copy(this._wristPrev[s]);
    }
  }

  /**
   * @param {{world:Float32Array, norm:Float32Array, vis:Float32Array}} s
   * @param pose BodyPose to fill
   */
  /**
   * 2D → 3D lifting for trackers without depth (MoveNet). Classic
   * bone-length trick: a limb that looks shorter than it really is must be
   * pointing toward the camera, by exactly sqrt(L² − l²). Torso yaw comes from
   * how narrow the shoulders look. Output uses MediaPipe's world convention
   * (metres, x = image right, y = down, z = away from camera).
   */
  _lift(s, aspect, dt) {
    const n = s.norm;
    const vis = s.vis;
    const W = (this._liftW ||= new Float32Array(33 * 3));
    const U = (i) => n[i * 2] * aspect;
    const V = (i) => n[i * 2 + 1];
    const sw = Math.hypot(U(LM.L_SHOULDER) - U(LM.R_SHOULDER), V(LM.L_SHOULDER) - V(LM.R_SHOULDER));
    // shoulder-width reference: rises fast, decays slowly (turning the torso
    // narrows the shoulders temporarily — that must read as yaw, not distance)
    if (!this._sRef) this._sRef = sw;
    this._sRef += (sw - this._sRef) * (sw > this._sRef ? 0.5 : 1 - Math.exp(-dt * 0.5));
    const hipsVis = Math.min(vis[LM.L_HIP], vis[LM.R_HIP]) > 0.4;
    const smx = (U(LM.L_SHOULDER) + U(LM.R_SHOULDER)) / 2;
    const smy = (V(LM.L_SHOULDER) + V(LM.R_SHOULDER)) / 2;
    const hmx = (U(LM.L_HIP) + U(LM.R_HIP)) / 2;
    const hmy = (V(LM.L_HIP) + V(LM.R_HIP)) / 2;
    const th = Math.hypot(smx - hmx, smy - hmy);
    // metres per image unit: torso height (~0.5 m) is yaw-proof; else shoulders (~0.36 m)
    const m = hipsVis && th > 0.05 ? 0.5 / th : 0.36 / Math.max(this._sRef, 0.02);
    const ox = hipsVis ? hmx : smx;
    const oy = hipsVis ? hmy : smy + 0.5 / m;
    for (let i = 0; i < 33; i++) {
      W[i * 3] = (U(i) - ox) * m;
      W[i * 3 + 1] = (V(i) - oy) * m;
      W[i * 3 + 2] = 0;
    }
    W[LM.NOSE * 3 + 2] = -0.1;
    W[LM.L_EYE * 3 + 2] = W[LM.R_EYE * 3 + 2] = -0.08;

    const len2 = (a, b) => Math.hypot(W[a * 3] - W[b * 3], W[a * 3 + 1] - W[b * 3 + 1]);
    const L = (this._bone ||= { Lu: 0.3, Lf: 0.27, Ru: 0.3, Rf: 0.27 });
    const depth = (full, seen) => Math.sqrt(Math.max(0, full * full - seen * seen));
    const arm = (sh, el, wr, ku, kf) => {
      const lu = len2(sh, el);
      const lf = len2(el, wr);
      // learn the user's real segment lengths from fully sideways poses
      L[ku] = Math.max(L[ku] + (0.3 - L[ku]) * dt * 0.05, Math.min(lu, 0.42));
      L[kf] = Math.max(L[kf] + (0.27 - L[kf]) * dt * 0.05, Math.min(lf, 0.38));
      return [depth(L[ku], lu), depth(L[kf], lf)];
    };
    const [lzu, lzf] = arm(LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST, 'Lu', 'Lf');
    const [rzu, rzf] = arm(LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST, 'Ru', 'Rf');

    // torso yaw: magnitude from shoulder narrowing, direction from which arm reaches further
    const cos = Math.min(1, sw / Math.max(this._sRef, 1e-3));
    const half = 0.18 * Math.sqrt(1 - cos * cos);
    const sgn = lzu + lzf > rzu + rzf ? 1 : -1; // left arm reaching → left shoulder forward
    W[LM.L_SHOULDER * 3 + 2] = -half * sgn;
    W[LM.R_SHOULDER * 3 + 2] = half * sgn;
    W[LM.L_ELBOW * 3 + 2] = W[LM.L_SHOULDER * 3 + 2] - lzu;
    W[LM.L_WRIST * 3 + 2] = W[LM.L_ELBOW * 3 + 2] - lzf;
    W[LM.R_ELBOW * 3 + 2] = W[LM.R_SHOULDER * 3 + 2] - rzu;
    W[LM.R_WRIST * 3 + 2] = W[LM.R_ELBOW * 3 + 2] - rzf;
    s.world = W;
    s.lifted = true;
    return s;
  }

  /**
   * @param armsOnly  Only the arms are driven by the camera. The torso, hips,
   *   head and legs keep whatever the caller already put in `pose` (a steady
   *   procedural stance), and the user's arm pose is re-expressed relative to
   *   the fighter's chest, so body-tracking noise can't disturb anything.
   */
  update(s, pose, aspect, dt, armsOnly = false) {
    this.armsOnly = armsOnly;
    if (!s.world) s = this._lift(s, aspect, dt);
    const P = this.pts;
    const w = s.world;
    const vis = s.vis;
    wp(w, LM.L_SHOULDER, P.ls);
    wp(w, LM.R_SHOULDER, P.rs);
    wp(w, LM.L_ELBOW, P.le);
    wp(w, LM.R_ELBOW, P.re);
    wp(w, LM.L_WRIST, P.lw);
    wp(w, LM.R_WRIST, P.rw);
    wp(w, LM.L_HIP, P.lh);
    wp(w, LM.R_HIP, P.rh);
    wp(w, LM.NOSE, P.nose);

    // MediaPipe's monocular depth is known to be compressed (a straight arm
    // pointed at the camera reads as maybe half its real reach). Expand the
    // arms' depth relative to their shoulders so reaching toward the camera
    // becomes a full, 3D forward punch on the avatar.
    if (!s.lifted) {
      for (const [sh, el, wr] of [[P.ls, P.le, P.lw], [P.rs, P.re, P.rw]]) {
        const elZ = el.z;
        el.z = sh.z + (elZ - sh.z) * DEPTH_GAIN;
        wr.z = el.z + (wr.z - elZ) * DEPTH_GAIN;
      }
    }
    wp(w, LM.L_EAR, P.lear);
    wp(w, LM.R_EAR, P.rear);

    this.present = Math.min(vis[LM.L_SHOULDER], vis[LM.R_SHOULDER]);

    // --- Torso frame (the user's) --------------------------------------------
    const right = this._uRight.subVectors(P.rs, P.ls).normalize();
    const midS = _a.addVectors(P.ls, P.rs).multiplyScalar(0.5);
    const midH = _b.addVectors(P.lh, P.rh).multiplyScalar(0.5);
    const measuredUp = _c.subVectors(midS, midH).normalize();
    // When hips are out of frame their estimate is guesswork: lean on a
    // vertical spine that only inherits the shoulder roll.
    const hipVis = Math.min(vis[LM.L_HIP], vis[LM.R_HIP]);
    _vert.set(0, 1, 0).addScaledVector(right, -right.y).normalize();
    this._uUp.copy(_vert).lerp(measuredUp, smooth01(0.3, 0.8, hipVis) * 0.85).normalize();
    if (!armsOnly) {
      pose.torsoRight.copy(this._uRight);
      pose.torsoUp.copy(this._uUp);
      pose.hipsShare = 0.3;
    }
    // the fighter's chest frame (arms-only mode maps the user's arms onto it)
    this._ax.copy(pose.torsoRight).multiplyScalar(-1).normalize();
    this._ay.copy(pose.torsoUp).addScaledVector(this._ax, -pose.torsoUp.dot(this._ax)).normalize();
    this._az.crossVectors(this._ax, this._ay);

    // --- Head ---------------------------------------------------------------
    const earVis = Math.min(vis[LM.L_EAR], vis[LM.R_EAR], vis[LM.NOSE]);
    if (armsOnly) {
      // head stays with the procedural stance
    } else if (earVis > 0.4) {
      this._headRight.subVectors(P.rear, P.lear).normalize();
      const midEar = _a.addVectors(P.lear, P.rear).multiplyScalar(0.5);
      const fwd = _b.subVectors(P.nose, midEar);
      // nose sits below the ear line: remove that bias (~14°)
      fwd.applyAxisAngle(this._headRight, 0.24).normalize();
      this._headUp.crossVectors(this._headRight, fwd).normalize();
      // blend toward the torso to damp the noisiest landmarks
      this._headUp.lerp(this._uUp, 0.3).normalize();
      pose.head = this.head;
    } else {
      pose.head = null;
    }

    // --- Arms ---------------------------------------------------------------
    // Chest frame: hand motion is measured relative to the torso, so turning
    // the body for one punch doesn't make the *other* hand look like it punched.
    this._bx.copy(this._uRight).multiplyScalar(-1).normalize(); // user's left
    this._by.copy(this._uUp).addScaledVector(this._bx, -this._uUp.dot(this._bx)).normalize();
    this._bz.crossVectors(this._bx, this._by); // forward
    this._arm('Left', pose.arms.Left, P.ls, P.le, P.lw, Math.min(vis[LM.L_ELBOW], vis[LM.L_WRIST]), dt);
    this._arm('Right', pose.arms.Right, P.rs, P.re, P.rw, Math.min(vis[LM.R_ELBOW], vis[LM.R_WRIST]), dt);

    // --- Punch mechanics: rotate the shoulder in, lean and step into the punch.
    // This is where real reach comes from (a straight arm alone falls short).
    let yaw = 0;
    let lean = 0;
    let step = 0;
    for (const side of ['Left', 'Right']) {
      const w = Math.max(pose.arms[side].assist || 0, pose.arms[side].elbowW || 0);
      if (!w) continue;
      const lead = (side === 'Left') === (this.fighter.stanceSign > 0);
      const kind = this.punch[side].kind;
      const turn = kind === 'elbow' ? 0.9 : kind === 'hook' || kind === 'bodyhook' || kind === 'overhand' ? 0.65 : lead ? 0.35 : 0.55;
      yaw += (side === 'Right' ? 1 : -1) * turn * w;
      lean += (kind === 'upper' ? 0.05 : kind === 'overhand' || kind === 'body' || kind === 'bodyhook' ? 0.2 : 0.15) * w;
      // step into the punch — further when the target is further away
      // (arm + shoulder turn + lean reach ≈ 0.72 m from the feet)
      // (elbows are short range: the step has to be bigger)
      const reach = kind === 'elbow' ? 0.42 : 0.72;
      const need = THREE.MathUtils.clamp(pose.arms[side].target.z - reach, 0.06, kind === 'elbow' ? 0.5 : 0.26);
      step = Math.max(step, need * w);
    }
    if (yaw) pose.torsoRight.applyAxisAngle(_vert.set(0, 1, 0), yaw);
    if (lean) pose.torsoUp.applyAxisAngle(_vert.set(1, 0, 0), lean).normalize();
    this.punchStep = step;

    if (armsOnly) {
      // stance is procedural: only add the step into a committed punch
      this._stepSm = (this._stepSm || 0) + ((this.punchStep || 0) - (this._stepSm || 0)) * (1 - Math.exp(-dt * 30));
      pose.hips.z += this._stepSm;
      this.advance = 0;
      return;
    }

    // --- Root motion from the image: slips, ducks, stepping -------------------
    if (this.calib) {
      const n = s.norm;
      const sx = (n[LM.L_SHOULDER * 2] + n[LM.R_SHOULDER * 2]) / 2;
      const sy = (n[LM.L_SHOULDER * 2 + 1] + n[LM.R_SHOULDER * 2 + 1]) / 2;
      const scale = this.fighter.rig.shoulderWidth / this.calib.sw;
      // user moving to their left → image right → character's left (+X)
      const x = THREE.MathUtils.clamp((sx - this.calib.sx) * aspect * scale * 0.85, -0.4, 0.4);
      const y = THREE.MathUtils.clamp(-(sy - this.calib.sy) * scale * 0.9, -0.38, 0.06);
      this.hipsTarget.set(x, y, 0);
    }
    this.hipsTarget.z = this.punchStep || 0;

    // --- Walking: closer to the camera = forward, further = back ------------------
    if (this.calib) {
      const c = this.calib.size;
      const now = bodySize(s, aspect);
      // average the two cues when both exist: each cancels the other's artefacts
      let ratio = c.sw > 0 && now.sw > 0 ? now.sw / c.sw : 1;
      if (c.th > 0 && now.th > 0) ratio = now.sw > 0 && c.sw > 0 ? (ratio + now.th / c.th) / 2 : now.th / c.th;
      this._ratio += (ratio - this._ratio) * (1 - Math.exp(-dt * 9));
      // pinhole camera: distance ∝ 1/size
      let adv = CALIB_DISTANCE * (1 - 1 / Math.max(0.3, this._ratio));
      if (Math.abs(adv) < 0.05) adv = 0; // dead-zone swallows tracking noise
      else adv -= Math.sign(adv) * 0.05;
      this.advance = THREE.MathUtils.clamp(adv * 1.3, -1.6, 1.4);
    }
    // light smoothing (the landmarks are already filtered+predicted)
    pose.hips.lerp(this.hipsTarget, 1 - Math.exp(-dt * 30));
    pose.bounce = 0;
  }

  /** Direction in the user's chest frame → same direction in the fighter's chest frame. */
  _remap(d) {
    const x = d.dot(this._bx);
    const y = d.dot(this._by);
    const z = d.dot(this._bz);
    return d.set(0, 0, 0).addScaledVector(this._ax, x).addScaledVector(this._ay, y).addScaledVector(this._az, z).normalize();
  }

  _arm(side, arm, sh, el, wr, v, dt) {
    const st = this.punch[side];
    const guard = this.guardPose[side];
    // Fast punches blur, which drops landmark confidence exactly when it
    // matters. Keep the last good direction instead of snapping to guard;
    // only drift to guard if the arm stays lost for a while.
    if (v > 0.12) {
      arm.upper.subVectors(el, sh).normalize();
      arm.fore.subVectors(wr, el).normalize();
      if (this.armsOnly) {
        // user chest frame → fighter chest frame
        this._remap(arm.upper);
        this._remap(arm.fore);
      }
      st.lostFor = 0;
    } else {
      st.lostFor += dt;
      if (st.lostFor > 0.5) {
        const k = 1 - Math.exp(-dt * 4);
        arm.upper.lerp(guard.upper, k).normalize();
        arm.fore.lerp(guard.fore, k).normalize();
      }
    }

    // --- Extension (0 = fist at shoulder, 1 = arm straight) -------------------
    const L = this.fighter.rig.upperArmLen;
    const F = this.fighter.rig.foreArmLen;
    _a.copy(arm.upper).multiplyScalar(L).addScaledVector(arm.fore, F);
    const e = _a.length() / (L + F);
    st.rate = dt > 0 ? (e - st.prevE) / dt : 0;
    st.prevE = st.e = e;
    // where the fist points from the shoulder, in the chest frame
    const aimLen = _a.length() || 1;
    const aimFwd = _a.dot(this._bz) / aimLen;
    const aimUp = _a.dot(this._by) / aimLen;

    // --- Fist position/velocity in the CHEST frame (m, m/s) ---------------------
    _c.subVectors(wr, sh);
    const rel = _b.set(_c.dot(this._bx), _c.dot(this._by), _c.dot(this._bz));
    if (dt > 0) this._wristVel[side].lerp(_c.subVectors(rel, this._wristPrev[side]).divideScalar(dt), 0.6);
    this._wristPrev[side].copy(rel);
    const vel = this._wristVel[side];
    const speed = vel.length();
    const inward = side === 'Left' ? -vel.x : vel.x; // toward the body's midline
    // elbow point in the chest frame (for elbow strikes)
    _c.subVectors(el, sh);
    const relE = this._relE[side].set(_c.dot(this._bx), _c.dot(this._by), _c.dot(this._bz));
    if (dt > 0) this._elbowVel[side].lerp(_c.subVectors(relE, this._elbowPrevRel[side]).divideScalar(dt), 0.6);
    this._elbowPrevRel[side].copy(relE);
    const velE = this._elbowVel[side];
    const speedE = velE.length();
    const inwardE = side === 'Left' ? -velE.x : velE.x;
    const fistTucked = rel.length() < 0.33; // fist stays by the head in an elbow strike
    this._clock = (this._clock || 0) + (side === 'Left' ? dt : 0);
    st.hist.push({ t: this._clock, p: rel.clone() });
    while (st.hist.length > 10) st.hist.shift();

    // --- Punch state machine: idle → candidate → active -----------------------------
    // A candidate follows the user's arm 1:1 (no assist). It only becomes a
    // punch once it proves itself: full fast extension aimed forward
    // (straight), or a long fast arc (hook/uppercut). Twitches never confirm.
    const P = PUNCH;
    if (st.phase === 'idle') {
      st.cool = Math.max(0, st.cool - dt);
      const straightOnset =
        st.rate > P.onsetRate && speed > P.onsetSpeed && vel.z > P.onsetFwd && vel.z > -vel.y * 0.5 && e > st.eRest + 0.08;
      const whipOnset = speed > P.whipSpeed && (inward > P.whipDir || vel.y > P.whipDir);
      // elbow: fist tucked, elbow raised to shoulder height, driving across/forward fast
      const elbowOnset = speedE > 2.0 && fistTucked && relE.y > -0.14 && (inwardE > 1.2 || velE.z > 1.2);
      if (st.cool <= 0 && (straightOnset || whipOnset || elbowOnset)) {
        st.originE.copy(relE);
        st.pathE = 0;
        st.phase = 'candidate';
        st.t = 0;
        st.maxE = e;
        // the punch began a few frames before it was fast enough to notice:
        // its start is the wound-up position — the recent point (≤0.25 s)
        // furthest from where the fist is now
        let i = st.hist.length - 1;
        let far = 0;
        for (let k = st.hist.length - 1; k >= 0 && this._clock - st.hist[k].t <= 0.25; k--) {
          const d = st.hist[k].p.distanceTo(rel);
          if (d > far) {
            far = d;
            i = k;
          }
        }
        st.origin.copy(st.hist[i].p);
        // direction of the throw: current motion (≈ the last 0.1 s) onwards
        st.sumV.copy(vel).multiplyScalar(0.1);
        st.path = 0;
        st.maxY = st.hist[i].p.y;
        for (let k = i + 1; k < st.hist.length; k++) {
          st.path += st.hist[k].p.distanceTo(st.hist[k - 1].p);
          st.maxY = Math.max(st.maxY, st.hist[k].p.y);
        }
      }
    }
    if (st.phase === 'candidate') {
      st.t += dt;
      st.path += speed * dt;
      st.maxE = Math.max(st.maxE, e);
      st.maxY = Math.max(st.maxY, rel.y);
      st.sumV.addScaledVector(vel, dt);
      st.pathE += speedE * dt;
      const elbowHit =
        st.pathE > 0.16 && speedE > 1.2 && fistTucked && relE.y > -0.1 && relE.z > 0.1 && relE.z > st.originE.z + 0.05;
      const D = _c.subVectors(rel, st.origin);
      // body shots are thrown with a slightly bent arm, aimed downward
      const needExt = aimUp < -0.3 ? P.confirmExt - 0.1 : P.confirmExt;
      // (a body shot travels forward AND down, so judge its total travel)
      const travelled = aimUp < -0.3 ? D.z > 0.15 && D.length() > 0.26 : D.z > P.confirmFwd;
      const straight =
        e > needExt && st.maxE > st.eRest + P.confirmGain - (aimUp < -0.3 ? 0.08 : 0) && aimFwd > P.confirmAim && travelled;
      // bent-arm punches must travel far, fast, and END in a striking zone:
      //  hook  → fist swept across toward the midline, out in front
      //  upper → fist driven up to above shoulder height, out in front
      // (raising the hands back to guard fails the zone test)
      // direction of the throw (motion since it started)
      const S = st.sumV;
      const lateral = side === 'Left' ? -S.x : S.x; // toward the midline
      const g = st.guardRel;
      // hooks and uppercuts must finish OUT IN FRONT of the guard — bringing the
      // arms back in (or up) to guard finishes at the guard, so it never counts
      const hookZone = lateral > 0.1 && lateral > Math.abs(S.y) && rel.z > g.z + 0.04;
      const upperZone = S.y > 0.1 && S.y > lateral && rel.y > g.y - 0.06 && rel.z > g.z + 0.05;
      // overhand: the fist loops UP over the guard, then drives forward and down
      const looped = st.maxY - st.origin.y > 0.1 && st.maxY - rel.y > 0.05;
      const overZone = looped && vel.y < -0.4 && rel.z > g.z + 0.06 && D.z > 0.1;
      const whip = st.path > P.confirmPath && speed > 1.5 && (hookZone || upperZone || overZone);
      if (elbowHit) {
        st.phase = 'active';
        st.active = true;
        st.id++;
        st.t = 0;
        st.kind = 'elbow';
      } else if (straight || whip) {
        st.phase = 'active';
        st.active = true;
        st.id++;
        st.t = 0;
        // classify: sideways travel dominates → hook, upward → uppercut
        if (overZone && (whip || straight)) st.kind = 'overhand';
        else if (whip && upperZone) st.kind = 'upper';
        else if (whip && hookZone && lateral > Math.abs(S.z) * 0.6) st.kind = rel.y < g.y - 0.2 ? 'bodyhook' : 'hook';
        else if (aimUp < -0.3 && rel.y < -0.15) st.kind = 'body';
        else st.kind = 'straight';
      } else if (st.t > P.candidateTime || (st.rate < -1 && speed < 1)) {
        st.phase = 'idle'; // never committed: it was just a movement
      }
    } else if (st.phase === 'active') {
      st.t += dt;
      st.path += speed * dt;
      const D = _c.subVectors(rel, st.origin);
      const travel = D.length() || 1e-6;
      let retracting = st.t > 0.08 && vel.dot(D) / travel < -0.9;
      if (st.kind === 'elbow') {
        st.pathE += speedE * dt;
        retracting = st.t > 0.08 && (velE.z < -0.8 || speedE < 0.5);
      }
      if (retracting || st.t > (st.kind === 'elbow' ? 0.6 : 0.8)) {
        st.phase = 'idle';
        st.active = false;
        st.cool = P.cooldown;
      }
    }

    // --- Assist: only for confirmed punches, and only for the final part ------------
    // The arm copies the user until the punch is committed; then the last
    // stretch is guided onto the target so the blow actually connects.
    let wIn = 0;
    if (st.phase === 'active' && st.kind === 'elbow') {
      wIn = smooth01(0.1, 0.28, st.pathE);
    } else if (st.phase === 'active') {
      const ext = smooth01(0.72, 0.95, e);
      const arc = smooth01(0.15, 0.35, st.path);
      wIn = st.kind === 'straight' || st.kind === 'body' ? Math.max(ext, arc * 0.8) : Math.max(arc, ext);
    }
    st.wOut = st.phase === 'active' ? Math.max(wIn, st.wOut) : Math.max(0, st.wOut - dt * 8);
    const w = st.wOut;
    arm.elbowW = 0;
    if (st.kind === 'elbow' && w > 0 && this.opponent) {
      // tracked arm + overlay driving the elbow point into the opponent's head
      arm.mode = 'dir';
      arm.assist = 0;
      arm.elbowW = w;
      this.fighter.toRootPos(_a.copy(this.opponent.headWorld), arm.elbowTarget);
      arm.elbowTarget.z += 0.1; // drive through the target
      arm.target.copy(arm.elbowTarget); // used for the step-in distance
      return;
    }
    arm.assist = w;
    if (w > 0 && this.opponent) {
      arm.mode = 'assist';
      const O = this.opponent;
      if (st.kind === 'body' || st.kind === 'bodyhook') _a.copy(O.bellyWorld).lerp(O.chestWorld, 0.4);
      else _a.copy(O.headWorld);
      this.fighter.toRootPos(_a, arm.target);
      if (st.kind === 'upper') arm.target.y -= 0.04;
      arm.target.z += 0.06; // follow through
      const out = side === 'Left' ? 1 : -1;
      if (st.kind === 'hook' || st.kind === 'bodyhook') arm.pole.set(out, 0.4, -0.3);
      else if (st.kind === 'upper') arm.pole.set(out * 0.4, -1, -0.2);
      else if (st.kind === 'overhand') arm.pole.set(out * 0.8, 0.9, -0.3);
      else arm.pole.copy(arm.upper); // keep the user's own elbow line
    } else {
      arm.mode = 'dir';
    }
  }
}
