// IK-driven boxing animation. Instead of canned clips, every punch is a glove
// trajectory solved with two-bone IK each frame, so punches can aim at a
// moving target (the other fighter's head) and blend freely with guards,
// slips and hit reactions. Used by the AI opponent and the keyboard mode.
import * as THREE from 'three';

const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const _v = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _ctrl = new THREE.Vector3();

// hand: 'lead' | 'rear'; path: straight | hook | upper
export const MOVES = {
  jab: { hand: 'lead', windup: 0.05, strike: 0.11, hold: 0.05, recover: 0.17, yaw: -0.22, lean: 0.06, path: 'straight', power: 0.75, label: 'JAB' },
  cross: { hand: 'rear', windup: 0.08, strike: 0.13, hold: 0.06, recover: 0.22, yaw: 0.55, lean: 0.1, path: 'straight', power: 1.0, label: 'CROSS' },
  hook: { hand: 'lead', windup: 0.1, strike: 0.15, hold: 0.06, recover: 0.24, yaw: -0.65, lean: 0.03, path: 'hook', power: 1.15, label: 'HOOK' },
  rhook: { hand: 'rear', windup: 0.11, strike: 0.15, hold: 0.06, recover: 0.25, yaw: 0.7, lean: 0.03, path: 'hook', power: 1.2, label: 'HOOK' },
  uppercut: { hand: 'rear', windup: 0.12, strike: 0.14, hold: 0.07, recover: 0.26, yaw: 0.35, lean: -0.02, path: 'upper', power: 1.3, crouch: 0.1, label: 'UPPERCUT' },
  body: { hand: 'rear', windup: 0.08, strike: 0.13, hold: 0.06, recover: 0.22, yaw: 0.45, lean: 0.18, path: 'straight', power: 0.9, crouch: 0.14, body: true, label: 'BODY SHOT' },
  lupper: { hand: 'lead', windup: 0.1, strike: 0.13, hold: 0.06, recover: 0.24, yaw: -0.3, lean: -0.02, path: 'upper', power: 1.15, crouch: 0.08, label: 'LEAD UPPERCUT' },
  overhand: { hand: 'rear', windup: 0.13, strike: 0.17, hold: 0.06, recover: 0.28, yaw: 0.75, lean: 0.16, path: 'overhand', power: 1.35, crouch: 0.04, label: 'OVERHAND' },
  bodyhook: { hand: 'lead', windup: 0.1, strike: 0.15, hold: 0.06, recover: 0.25, yaw: -0.6, lean: 0.1, path: 'hook', power: 1.1, crouch: 0.16, body: true, label: 'LIVER SHOT' },
  // elbows: short range — the torso whips round and the folded arm drives the elbow in
  lelbow: { hand: 'lead', windup: 0.12, strike: 0.14, hold: 0.07, recover: 0.24, yaw: -0.95, lean: 0.14, path: 'elbow', power: 1.35, elbow: true, label: 'LEAD ELBOW' },
  relbow: { hand: 'rear', windup: 0.12, strike: 0.14, hold: 0.07, recover: 0.24, yaw: 1.0, lean: 0.14, path: 'elbow', power: 1.45, elbow: true, label: 'REAR ELBOW' },
  // defensive swat: the lead glove slaps the incoming punch off line
  parry: { hand: 'lead', windup: 0.01, strike: 0.07, hold: 0.07, recover: 0.12, yaw: -0.08, lean: 0, path: 'straight', power: 0, parry: true, label: 'PARRY' },
};

export class ProceduralBoxer {
  constructor(fighter) {
    this.fighter = fighter;
    const rig = fighter.rig;
    this.headY = rig.headRest.y;
    this.sign = fighter.stanceSign; // orthodox: lead = Left
    this.lead = this.sign > 0 ? 'Left' : 'Right';
    this.rear = this.sign > 0 ? 'Right' : 'Left';
    this.baseYaw = -0.32 * this.sign;

    this.guard = {
      [this.lead]: new THREE.Vector3(0.1 * this.sign, this.headY - 0.1, 0.3),
      [this.rear]: new THREE.Vector3(-0.12 * this.sign, this.headY - 0.1, 0.19),
    };
    this.highGuard = {
      [this.lead]: new THREE.Vector3(0.075 * this.sign, this.headY + 0.02, 0.2),
      [this.rear]: new THREE.Vector3(-0.075 * this.sign, this.headY + 0.02, 0.17),
    };
    this.lowArms = false;

    this.glove = { Left: this.guard.Left.clone(), Right: this.guard.Right.clone() };
    this.action = null; // current punch
    this.queue = [];
    this.blockAmt = 0;
    this.blockTarget = 0;
    this.yaw = this.baseYaw;
    this.lean = 0;
    this.crouch = 0;
    this.slip = 0; // lateral offset target
    this.duck = 0;
    this.hipsCur = new THREE.Vector3();
    this.t = Math.random() * 10;
    this.idleAmp = 1;
    this.onImpact = null; // (action) => void, fired at full extension
    this.onLaunch = null;
    this.knockedOut = 0; // 0..1 knockdown blend
    this.telegraph = 0; // 0..1 how visible the windup is (AI tells)
    this.elbowAmt = 0; // 0..1 elbow-strike overlay on the active arm
    this.elbowPoint = new THREE.Vector3();
  }

  get busy() {
    return !!this.action || this.queue.length > 0;
  }

  /** Queue a punch. getTarget() returns the root-space aim point. */
  punch(name, getTarget, opts = {}) {
    const m = MOVES[name];
    if (!m) return;
    this.queue.push({ name, m, getTarget, windup: opts.windup ?? m.windup, id: Math.random() });
  }

  cancel() {
    this.queue.length = 0;
    if (this.action && this.action.phase !== 'recover') {
      this.action.phase = 'recover';
      this.action.t = 0;
      this.action.from.copy(this.glove[this.action.side]);
    }
  }

  _start(item) {
    const { m } = item;
    const side = m.hand === 'lead' ? this.lead : this.rear;
    this.action = {
      ...item,
      side,
      phase: 'windup',
      t: 0,
      from: this.glove[side].clone(),
      target: new THREE.Vector3(),
      hitDone: false,
    };
  }

  update(dt, pose) {
    this.t += dt;
    const A = this.action;
    if (!A && this.queue.length) this._start(this.queue.shift());

    // --- guard / idle motion --------------------------------------------------
    this.blockAmt += (this.blockTarget - this.blockAmt) * (1 - Math.exp(-dt * 22));
    const bob = Math.sin(this.t * 2.1) * 0.035 * this.idleAmp;
    const weave = Math.sin(this.t * 1.3 + 0.7) * 0.05 * this.idleAmp;

    let yaw = this.baseYaw;
    let lean = 0.04;
    let crouch = 0;
    let step = 0; // step into the punch when the target is at the edge of range
    const pole = { Left: pose.arms.Left.pole, Right: pose.arms.Right.pole };

    for (const s of ['Left', 'Right']) {
      const out = s === 'Left' ? 1 : -1;
      const g = _v.lerpVectors(this.guard[s], this.highGuard[s], this.blockAmt);
      g.y += bob * 0.5;
      if (this.lowArms) g.set(0.18 * out, this.headY - 0.55, 0.12);
      this.glove[s].lerp(g, 1 - Math.exp(-dt * 16));
      pole[s].set(out * 0.9, -1, -0.5);
    }

    // --- active punch ------------------------------------------------------
    if (A) {
      const m = A.m;
      const s = A.side;
      const mYaw = m.yaw * this.sign; // move yaws are authored for orthodox; mirror for southpaw
      const out = s === 'Left' ? 1 : -1;
      A.t += dt;
      const dur = A.phase === 'windup' ? A.windup : m[A.phase];
      let k = Math.min(1, A.t / dur);
      const glove = this.glove[s];
      const guardPos = this.guard[s];
      if (A.phase === 'windup') {
        // pull back / load up: this is the readable "tell"
        _v.copy(guardPos);
        if (m.path === 'elbow') _v.set(out * 0.13, this.headY + 0.03, 0.1); // fist by the ear
        else if (m.path === 'hook') _v.add(_v2.set(out * 0.12, 0.02, -0.06));
        else if (m.path === 'overhand') _v.add(_v2.set(out * 0.1, 0.1, -0.1));
        else if (m.path === 'upper') _v.add(_v2.set(-out * 0.02, -0.22, -0.02));
        else _v.add(_v2.set(0, -0.02, -0.07));
        glove.lerpVectors(A.from, _v, easeInOut(k));
        yaw += (mYaw - this.baseYaw) * -0.25 * k;
        crouch += (m.crouch || 0) * k;
        this.telegraph = k;
        if (k >= 1) {
          A.phase = 'strike';
          A.t = 0;
          A.from.copy(glove);
          A.getTarget(A.target);
          this.onLaunch?.(A);
        }
      } else if (A.phase === 'strike') {
        A.getTarget(_v2);
        // weak homing (you can still slip a punch); elbows are thrown in close
        // and follow the head much more
        A.target.lerp(_v2, 1 - Math.exp(-dt * (m.elbow ? 12 : 3)));
        const e = easeOut(k);
        if (m.path === 'hook') {
          _ctrl.set(A.target.x + out * 0.42, A.target.y + 0.02, (A.from.z + A.target.z) * 0.5);
          bezier(A.from, _ctrl, A.target, e, glove);
          pole[s].set(out * 1, 0.35, -0.25);
        } else if (m.path === 'upper') {
          _ctrl.set(A.target.x * 0.6, A.target.y - 0.35, A.target.z * 0.7);
          bezier(A.from, _ctrl, A.target, e, glove);
          pole[s].set(out * 0.4, -1, -0.2);
        } else if (m.path === 'elbow') {
          // fist stays tucked by the head; the elbow sweeps across into the target
          glove.set(out * 0.1 - out * 0.08 * e, this.headY + 0.02, 0.12);
          pole[s].set(out, 0.3, -0.4);
        } else if (m.path === 'overhand') {
          // loops up and over the guard, coming down onto the head
          _ctrl.set(A.target.x + out * 0.18, A.target.y + 0.38, (A.from.z + A.target.z) * 0.5);
          bezier(A.from, _ctrl, A.target, e, glove);
          pole[s].set(out * 0.8, 0.9, -0.3);
        } else {
          glove.lerpVectors(A.from, A.target, e);
          pole[s].set(out * 0.7, -0.6, -0.3);
        }
        yaw = this.baseYaw + (mYaw - this.baseYaw) * e;
        lean += m.lean * e;
        step = THREE.MathUtils.clamp(A.target.z - (m.elbow ? 0.36 : 0.72), 0.04, m.elbow ? 0.5 : 0.28) * e;
        crouch += (m.crouch || 0) * (m.path === 'upper' ? 1 - e : 1);
        this.telegraph = 1 - e;
        if (k >= 1) {
          A.phase = 'hold';
          A.t = 0;
          this.onImpact?.(A);
        }
      } else if (A.phase === 'hold') {
        glove.copy(A.target);
        yaw = mYaw;
        lean += m.lean;
        step = THREE.MathUtils.clamp(A.target.z - (m.elbow ? 0.36 : 0.72), 0.04, m.elbow ? 0.5 : 0.28);
        crouch += m.path === 'upper' ? 0 : m.crouch || 0;
        pole[s].set(out * (m.path === 'hook' ? 1 : 0.7), m.path === 'hook' ? 0.35 : m.path === 'overhand' ? 0.9 : -0.6, -0.3);
        if (k >= 1) {
          A.phase = 'recover';
          A.t = 0;
          A.from.copy(glove);
        }
      } else if (A.phase === 'recover') {
        const e = easeInOut(k);
        glove.lerpVectors(A.from, guardPos, e);
        yaw = mYaw + (this.baseYaw - mYaw) * e;
        lean += m.lean * (1 - e);
        step = THREE.MathUtils.clamp(A.target.z - (m.elbow ? 0.36 : 0.72), 0.04, m.elbow ? 0.5 : 0.28) * (1 - e);
        crouch += (m.path === 'upper' ? 0 : m.crouch || 0) * (1 - e);
        if (k >= 1) {
          this.action = null;
          this.telegraph = 0;
        }
      }
    }

    // --- elbow overlay ------------------------------------------------------------
    let elbowW = 0;
    if (A && A.m.elbow && this.action === A) {
      const dur = A.phase === 'windup' ? A.windup : A.m[A.phase];
      const k = Math.min(1, A.t / dur);
      const out = A.side === 'Left' ? 1 : -1;
      // elbow path: raised out to the side → across into the target
      _v.set(out * 0.4, this.headY - 0.1, 0.1);
      if (A.phase === 'windup') {
        elbowW = 0.35 * easeInOut(k);
        this.elbowPoint.copy(_v);
      } else if (A.phase === 'strike') {
        const e = easeOut(k);
        elbowW = 0.35 + 0.65 * e;
        this.elbowPoint.lerpVectors(_v, A.target, e);
      } else if (A.phase === 'hold') {
        elbowW = 1;
        this.elbowPoint.copy(A.target);
      } else {
        elbowW = 1 - easeInOut(k);
      }
    }
    this.elbowAmt += (elbowW - this.elbowAmt) * (1 - Math.exp(-dt * 30));

    // --- write BodyPose --------------------------------------------------------
    this.yaw += (yaw - this.yaw) * (1 - Math.exp(-dt * 30));
    this.lean += (lean - this.lean) * (1 - Math.exp(-dt * 20));
    this.crouch += (crouch - this.crouch) * (1 - Math.exp(-dt * 18));

    const hipsTarget = _v.set(weave + this.slip, -this.crouch - this.duck - Math.abs(weave) * 0.4, step);
    this.hipsCur.lerp(hipsTarget, 1 - Math.exp(-dt * 14));
    pose.hips.copy(this.hipsCur);

    // torso frame: yaw about Y, forward lean about the character's right axis,
    // lateral tilt that follows slips (a real "slip" is a torso lean)
    const tilt = -(weave + this.slip) * 0.9;
    pose.torsoUp.set(0, 1, 0);
    pose.torsoUp.applyAxisAngle(_v2.set(1, 0, 0), this.lean + this.duck * 0.8);
    pose.torsoUp.applyAxisAngle(_v2.set(0, 0, 1), tilt);
    pose.torsoRight.set(-1, 0, 0).applyAxisAngle(_v2.set(0, 1, 0), this.yaw);
    pose.hipsShare = 0.45;
    pose.head = null;
    pose.headPitch = 0.08 - this.lean * 0.5 - this.duck * 0.4;
    pose.bounce = 0.018 * this.idleAmp;

    for (const s of ['Left', 'Right']) {
      const arm = pose.arms[s];
      arm.mode = 'ik';
      const striking = this.action && this.action.m.elbow && this.action.side === s;
      arm.elbowW = striking ? this.elbowAmt : 0;
      if (striking) arm.elbowTarget.copy(this.elbowPoint);
      arm.target.copy(this.glove[s]);
      // gloves travel with the body when it slips or ducks
      arm.target.x += this.hipsCur.x * 0.85;
      arm.target.y += this.hipsCur.y * 0.9;
    }

    // --- knockdown overrides everything ------------------------------------------
    if (this.knockedOut > 0) {
      const k = this.knockedOut;
      pose.hips.y -= 0.45 * k;
      pose.torsoUp.lerp(_v2.set(0, 0.55, 0.85), k).normalize();
      pose.headPitch += 0.5 * k;
      pose.bounce *= 1 - k;
      for (const s of ['Left', 'Right']) {
        const out = s === 'Left' ? 1 : -1;
        pose.arms[s].target.lerp(_v2.set(0.22 * out, 0.35, 0.35), k);
        pose.arms[s].pole.set(out, -0.3, -1);
      }
    }
  }
}

function bezier(a, c, b, t, out) {
  const u = 1 - t;
  return out.set(
    u * u * a.x + 2 * u * t * c.x + t * t * b.x,
    u * u * a.y + 2 * u * t * c.y + t * t * b.y,
    u * u * a.z + 2 * u * t * c.z + t * t * b.z,
  );
}
