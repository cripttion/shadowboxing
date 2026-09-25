// A 3D boxer: skinned avatar + gloves + full-body solver.
// Drivers (camera / AI / keyboard) hand it a BodyPose each frame; the fighter
// turns that into bone rotations, plants its feet with leg IK, and layers
// physically-springy hit reactions on top.
import * as THREE from 'three';
import { clone as cloneSkinned } from 'three/examples/jsm/utils/SkeletonUtils.js';
import { Rig, frameRotation, UP } from './Rig.js';
import { createGlove } from './Glove.js';
import { applyKit } from './Outfit.js';

const _v = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _v4 = new THREE.Vector3();
const _up = new THREE.Vector3();
const _lo = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _qT = new THREE.Quaternion();
const _qH = new THREE.Quaternion();
const _qS = new THREE.Quaternion();
const _qHead = new THREE.Quaternion();
const _qYaw = new THREE.Quaternion();

const SIDES = ['Left', 'Right'];
const FINGERS = ['Index', 'Middle', 'Ring', 'Pinky'];

const newArm = () => ({
  mode: 'ik',
  upper: new THREE.Vector3(),
  fore: new THREE.Vector3(),
  target: new THREE.Vector3(),
  pole: new THREE.Vector3(),
  // elbow strike overlay: 0 = normal arm, 1 = folded arm driving the elbow at elbowTarget
  elbowW: 0,
  elbowTarget: new THREE.Vector3(),
});

export function createBodyPose() {
  return {
    hips: new THREE.Vector3(), // root-space offset from the stance position
    torsoUp: new THREE.Vector3(0, 1, 0),
    torsoRight: new THREE.Vector3(-1, 0, 0),
    hipsShare: 0.35,
    head: null, // {up, right} or null to follow torso
    headPitch: 0,
    bounce: 0,
    arms: {
      Left: newArm(),
      Right: newArm(),
    },
  };
}

class Spring3 {
  constructor(k = 140, d = 13) {
    this.x = new THREE.Vector3();
    this.v = new THREE.Vector3();
    this.k = k;
    this.d = d;
  }
  kick(impulse) {
    this.v.add(impulse);
  }
  step(dt) {
    // semi-implicit Euler, sub-stepped for stability at low frame rates
    const n = Math.ceil(dt / 0.008);
    const h = dt / n;
    for (let i = 0; i < n; i++) {
      this.v.addScaledVector(this.x, -this.k * h).multiplyScalar(Math.max(0, 1 - this.d * h));
      this.x.addScaledVector(this.v, h);
    }
  }
}

export class Fighter {
  /**
   * @param {THREE.Object3D} source loaded glTF scene (shared, will be cloned)
   */
  constructor(source, { gloveColor = 0xc81d25, kit = null, stance = 'orthodox', name = 'fighter' } = {}) {
    this.name = name;
    this.group = new THREE.Group();
    this.group.name = name;
    this.model = cloneSkinned(source);
    this.group.add(this.model);

    this.meshes = [];
    this.model.traverse((o) => {
      if (o.isMesh) {
        o.castShadow = true;
        o.receiveShadow = false;
        o.frustumCulled = false; // skinned bounds don't follow the pose
        // own materials so each fighter wears its own kit
        const mn = o.material.name || '';
        if (/Headwear/i.test(mn) || /Headwear/i.test(o.name)) {
          o.visible = false;
          o.userData.alwaysHidden = true; // hat + hair piece: never shown
        }
        if (kit) applyKit(o, kit);
        else o.material = o.material.clone();
        if (o.material.envMapIntensity !== undefined) o.material.envMapIntensity = 0.9;
        this.meshes.push(o);
      }
    });

    // Facial expressions: morph targets baked by tools/character/gen_boxer.py
    // on the body, brows, lashes and teeth (all must move together)
    this.faceMeshes = [];
    this.model.traverse((o) => {
      if (o.isMesh && o.morphTargetDictionary && Object.keys(o.morphTargetDictionary).length) this.faceMeshes.push(o);
    });
    this.expr = { focus: 0, effort: 0, pain: 0, exhale: 0, mouthOpen: 0, blink: 0, ko: 0, smile: 0 };
    this.pain = 0; // decays after a hit
    this.exhale = 0; // short "tss" breath on each punch
    this.effortT = 0; // strain while throwing a punch
    this.focusLevel = 0.5; // resting "game face" (set by the game)
    this.koLevel = 0; // knocked out: eyes shut, jaw slack
    this.smileLevel = 0; // winner's smile
    this.blinkT = 0;
    this.nextBlink = 1 + Math.random() * 3;

    this.rig = new Rig(this.group);
    const bones = this.rig.bones;

    // Tuck the fingers into the glove
    for (const s of SIDES) {
      for (const f of FINGERS) bones[`${s}Hand${f}1`]?.scale.setScalar(0.35);
      bones[`${s}HandThumb1`]?.scale.setScalar(0.45);
    }

    // Gloves
    this.gloves = {};
    for (const s of SIDES) {
      const hand = bones[s + 'Hand'];
      const glove = createGlove(gloveColor);
      const fingerDir = bones[s + 'HandMiddle1'].position.clone().normalize();
      const thumbDir = bones[s + 'HandThumb1'].position.clone().normalize();
      thumbDir.addScaledVector(fingerDir, -thumbDir.dot(fingerDir)).normalize();
      const y = new THREE.Vector3().crossVectors(fingerDir, thumbDir);
      glove.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(thumbDir, y, fingerDir));
      glove.position.copy(fingerDir).multiplyScalar(0.075);
      hand.add(glove);
      this.gloves[s] = glove;
    }

    // Stance: orthodox = left foot forward
    const sign = stance === 'orthodox' ? 1 : -1;
    this.stanceSign = sign;
    const ankleY = this.rig.restPos('LeftFoot').y;
    this.ankleY = ankleY;
    this.feet = {};
    for (const s of SIDES) {
      const lead = (s === 'Left') === (sign > 0);
      const base = new THREE.Vector3(s === 'Left' ? 0.13 : -0.13, 0, lead ? 0.15 : -0.13);
      const yaw = lead ? -0.1 * sign : -0.65 * sign;
      this.feet[s] = {
        base,
        yaw,
        anchor: base.clone(),
        from: base.clone(),
        to: base.clone(),
        t: 1,
        pole: new THREE.Vector3(s === 'Left' ? 0.35 : -0.35, 0, 1),
      };
    }
    this.baseCrouch = 0.07;
    this.fatigue = 0; // 0..1, set by the game from remaining health
    this.rootVel = new THREE.Vector3(); // root-space velocity of the whole fighter
    this.time = 0;

    // Hit reaction springs (axis-angle vectors in root space)
    this.headKick = new Spring3(160, 11);
    this.bodyKick = new Spring3(90, 10);
    this.push = new Spring3(40, 9); // positional recoil (root space)

    // Temp storage reused per frame
    this._ikU = new THREE.Vector3();
    this._ikL = new THREE.Vector3();
    this._eu = new THREE.Vector3();
    this._ef = new THREE.Vector3();
    this._bu = new THREE.Vector3();
    this._bf = new THREE.Vector3();
    this.headWorld = new THREE.Vector3();
    this.chestWorld = new THREE.Vector3();
    this.bellyWorld = new THREE.Vector3();
    this.gloveWorld = { Left: new THREE.Vector3(), Right: new THREE.Vector3() };
    this.gloveVel = { Left: new THREE.Vector3(), Right: new THREE.Vector3() };
    this.elbowWorld = { Left: new THREE.Vector3(), Right: new THREE.Vector3() };
    this.elbowVel = { Left: new THREE.Vector3(), Right: new THREE.Vector3() };
    this._elbowPrev = { Left: new THREE.Vector3(), Right: new THREE.Vector3() };
    this._glovePrev = { Left: new THREE.Vector3(), Right: new THREE.Vector3() };
    this._first = true;
  }

  /** First-person view hides the head (and face parts) of this fighter. */
  setHeadVisible(on) {
    if (this._headOn === on) return;
    this._headOn = on;
    // collapse the head bone: the skin around the eyes/mouth would otherwise
    // be seen from inside the skull in first person
    this.rig.bones.Head.scale.setScalar(on ? 1 : 0.001);
    for (const m of this.meshes) {
      if (m.userData.alwaysHidden) continue;
      // face parts near the FPP eye point (the head itself is back-face culled from inside)
      if (/Wolf3D_Head$|Beard|Teeth|Eye|Hair|Glasses|low-poly|eyebrow|eyelash|short\d/i.test(m.name)) m.visible = on;
    }
  }

  get bones() {
    return this.rig.bones;
  }

  /** Boxers exhale sharply on every punch. */
  breathe(amount = 1) {
    this.exhale = Math.max(this.exhale, 0.55 * Math.min(1.3, amount));
  }

  /** Strain on the face while throwing a punch. */
  effort(amount = 1) {
    this.effortT = Math.max(this.effortT, Math.min(1, 0.7 + 0.3 * amount));
  }

  hit(dirRoot, strength, zone = 'head') {
    this.pain = Math.min(1, this.pain + 0.5 + strength * 0.4);
    this.blinkT = 0.16; // flinch
    // rotate away from the incoming punch: axis = up × dir
    _v.crossVectors(UP, dirRoot).normalize();
    if (zone === 'head') {
      this.headKick.kick(_v.clone().multiplyScalar(strength * 9));
      this.headKick.kick(new THREE.Vector3(-strength * 4, 0, 0)); // chin up
      this.bodyKick.kick(_v.clone().multiplyScalar(strength * 2.5));
    } else {
      this.bodyKick.kick(new THREE.Vector3(strength * 5, 0, 0)); // fold forward
      this.bodyKick.kick(_v.multiplyScalar(strength * 2));
    }
    this.push.kick(_v2.copy(dirRoot).setY(0).normalize().multiplyScalar(strength * 1.2));
  }

  /** Apply a BodyPose. */
  solve(pose, dt) {
    this.time += dt;
    const rig = this.rig;
    const b = rig.bones;
    rig.begin();

    this.headKick.step(dt);
    this.bodyKick.step(dt);
    this.push.step(dt);

    // --- Hips -------------------------------------------------------------
    const bounce = pose.bounce * (0.5 + 0.5 * Math.sin(this.time * Math.PI * 2 * 1.7));
    _v.copy(rig.hipsRest).add(pose.hips).add(this.push.x);
    _v.y -= this.baseCrouch + bounce;
    rig.setSpacePos(b.Hips, _v);

    // --- Torso: match the shoulder frame, distribute along the spine --------
    _up.copy(pose.torsoUp).addScaledVector(this.push.x, -0.6).normalize();
    frameRotation(rig.restTorsoUp, rig.restRight, _up, pose.torsoRight, _qT);
    _q.setFromAxisAngle(_v2.copy(this.bodyKick.x).normalize(), this.bodyKick.x.length());
    _qT.premultiply(_q);
    _qH.identity().slerp(_qT, pose.hipsShare);
    rig.setFrame('Hips', _qH);
    const spine = ['Spine', 'Spine1', 'Spine2'];
    for (let i = 0; i < 3; i++) {
      _qS.slerpQuaternions(_qH, _qT, (i + 1) / 3);
      rig.setFrame(spine[i], _qS);
    }

    // --- Head -------------------------------------------------------------
    if (pose.head) {
      frameRotation(UP, rig.restRight, pose.head.up, pose.head.right, _qHead);
    } else {
      _qHead.copy(_qT);
    }
    if (pose.headPitch) {
      _q.setFromAxisAngle(_v2.set(1, 0, 0), pose.headPitch);
      _qHead.premultiply(_q);
    }
    _q.setFromAxisAngle(_v2.copy(this.headKick.x).normalize(), this.headKick.x.length());
    _qHead.premultiply(_q);
    _qS.slerpQuaternions(_qT, _qHead, 0.45);
    rig.setFrame('Neck', _qS);
    rig.setFrame('Head', _qHead);

    // --- Arms -------------------------------------------------------------
    for (const s of SIDES) {
      b[s + 'Shoulder'].quaternion.copy(rig.rest.get(b[s + 'Shoulder']).local);
      const arm = pose.arms[s];
      let U = arm.upper;
      let F = arm.fore;
      if (arm.mode === 'assist' || arm.mode === 'ik') {
        rig.pos(s + 'Arm', _v3);
        if (arm.mode === 'assist') {
          // tracked glove position (where the user's fist maps to)…
          _v4.copy(_v3)
            .addScaledVector(arm.upper, rig.upperArmLen)
            .addScaledVector(arm.fore, rig.foreArmLen + 0.09);
          // …pulled onto the target in proportion to the user's extension
          _v4.lerp(arm.target, arm.assist);
        } else {
          _v4.copy(arm.target);
        }
        // glove centre sits ~9cm past the wrist
        const dist = _v2.subVectors(_v4, _v3).length();
        _v4.copy(_v3).addScaledVector(_v2, Math.max(0, dist - 0.09) / (dist || 1));
        Rig.twoBone(_v3, _v4, rig.upperArmLen, rig.foreArmLen, arm.pole, this._ikU, this._ikL);
        U = this._ikU;
        F = this._ikL;
      }
      if (arm.elbowW > 0.001) {
        // Elbow strike: the upper arm drives the elbow point at the target
        // while the forearm folds back so the fist stays by your own head.
        rig.pos(s + 'Arm', _v3);
        const eu = this._eu.subVectors(arm.elbowTarget, _v3).normalize();
        const inward = s === 'Left' ? -1 : 1;
        const ef = this._ef.copy(eu).multiplyScalar(-0.8).add(_v2.set(inward * 0.55, 0.35, 0)).normalize();
        const w = Math.min(1, arm.elbowW);
        U = this._bu.copy(U).lerp(eu, w).normalize();
        F = this._bf.copy(F).lerp(ef, w).normalize();
      }
      rig.solveArm(s, U, F);
    }

    // --- Legs: planted feet + two-bone IK ------------------------------------
    this._solveLegs(pose, dt);

    this._updateFace(dt);

    // --- World-space probes for gameplay --------------------------------
    this._updateProbes(dt);
  }

  _solveLegs(pose, dt) {
    const rig = this.rig;
    const b = rig.bones;
    // Feet are planted in the WORLD: when the whole fighter travels (walking)
    // the anchors are shifted back so the feet stay on the canvas…
    this.group.getWorldPosition(_v);
    if (this._lastWorld) {
      _v2.subVectors(_v, this._lastWorld);
      if (_v2.lengthSq() > 0.25) {
        // teleport (reset): re-seat the feet instead of sliding them
        for (const s of SIDES) {
          const f = this.feet[s];
          f.anchor.copy(f.base);
          f.from.copy(f.base);
          f.to.copy(f.base);
          f.t = 1;
        }
        this.rootVel.set(0, 0, 0);
      } else {
        this.toRootDir(_v2, _v3);
        for (const s of SIDES) {
          const f = this.feet[s];
          f.anchor.sub(_v3);
          f.from.sub(_v3);
          f.to.sub(_v3);
        }
        if (dt > 0) this.rootVel.lerp(_v3.divideScalar(dt), 1 - Math.exp(-dt * 10));
      }
    }
    (this._lastWorld ||= new THREE.Vector3()).copy(_v);

    // …and step (one foot at a time, the one lagging most) when the body gets
    // away from them. Targets lead with velocity so steps land ahead, like a
    // real step-drag footwork pattern instead of feet dragging behind.
    const drift = _v.set(pose.hips.x + this.push.x.x, 0, pose.hips.z + this.push.x.z);
    _v4.copy(this.rootVel).setY(0).multiplyScalar(0.16).clampLength(0, 0.22);
    drift.add(_v4);
    let stepping = false;
    for (const s of SIDES) if (this.feet[s].t < 1) stepping = true;
    let pick = null;
    let worst = 0.09;
    for (const s of SIDES) {
      const f = this.feet[s];
      if (f.t < 1) {
        const speed = this.rootVel.length();
        f.t = Math.min(1, f.t + dt / (speed > 0.4 ? 0.14 : 0.18));
      } else if (!stepping) {
        const err = _v2.copy(f.base).add(drift).distanceTo(f.anchor);
        if (err > worst) {
          worst = err;
          pick = s;
        }
      }
    }
    if (pick) {
      const f = this.feet[pick];
      f.from.copy(f.anchor);
      f.to.copy(f.base).add(drift);
      f.t = 0;
    }
    for (const s of SIDES) {
      const f = this.feet[s];
      const e = f.t * f.t * (3 - 2 * f.t);
      f.anchor.lerpVectors(f.from, f.to, e);
      if (f.t >= 1) f.anchor.copy(f.to);
    }
    for (const s of SIDES) {
      const f = this.feet[s];
      const lift = f.t < 1 ? Math.sin(Math.PI * f.t) * 0.06 : 0;
      _v3.set(f.anchor.x, this.ankleY + lift, f.anchor.z);
      rig.pos(s + 'UpLeg', _v4);
      Rig.twoBone(_v4, _v3, rig.thighLen, rig.shinLen, f.pole, this._ikU, this._ikL);
      rig.aim(s + 'UpLeg', s + 'Leg', this._ikU);
      rig.aim(s + 'Leg', s + 'Foot', this._ikL);
      _qYaw.setFromAxisAngle(UP, f.yaw);
      _q.copy(rig.rest.get(b[s + 'Foot']).worldQ).premultiply(_qYaw);
      rig.setSpaceQ(b[s + 'Foot'], _q);
    }
  }

  /** Blend the facial expressions (pain > effort > focus; KO overrides all). */
  _updateFace(dt) {
    this.pain = Math.max(0, this.pain - dt * 1.4);
    this.exhale = Math.max(0, this.exhale - dt * 3.2);
    this.effortT = Math.max(0, this.effortT - dt * 2.2);

    // natural blinking (and a flinch-blink when hit)
    this.nextBlink -= dt;
    if (this.nextBlink <= 0 && this.blinkT <= 0) {
      this.blinkT = 0.16;
      this.nextBlink = 2 + Math.random() * 4;
    }
    let blink = 0;
    if (this.blinkT > 0) {
      this.blinkT -= dt;
      blink = Math.sin(Math.PI * Math.max(0, this.blinkT) / 0.16);
    }

    const ko = this.koLevel;
    const pain = this.pain * (1 - ko);
    const awake = (1 - ko) * (1 - pain * 0.85);
    const breath = this.fatigue ? (0.5 + 0.5 * Math.sin(this.time * 5.5)) * 0.45 * this.fatigue : 0;
    const target = this.expr;
    const want = {
      focus: this.focusLevel * awake * (1 - this.effortT * 0.6) * (1 - this.smileLevel),
      effort: this.effortT * awake,
      pain,
      exhale: this.exhale * (1 - ko) * (1 - pain * 0.5),
      mouthOpen: Math.max(breath * (1 - ko), 0) * (1 - this.exhale),
      blink: ko > 0.5 ? 0 : blink,
      ko,
      smile: this.smileLevel * (1 - ko) * (1 - pain),
    };
    for (const k in want) {
      const rate = k === 'blink' ? 1000 : k === 'pain' || k === 'exhale' ? 28 : 10;
      target[k] += (want[k] - target[k]) * (1 - Math.exp(-dt * rate));
    }
    for (const m of this.faceMeshes) {
      const dict = m.morphTargetDictionary;
      const inf = m.morphTargetInfluences;
      for (const k in target) {
        const i = dict[k];
        if (i !== undefined) inf[i] = target[k];
      }
    }
  }

  _updateProbes(dt) {
    const b = this.rig.bones;
    this.group.updateMatrixWorld(true);
    // head centre ≈ between Head joint and skull top, nudged to the face
    b.Head.getWorldPosition(this.headWorld);
    if (b.HeadTop_End) {
      b.HeadTop_End.getWorldPosition(_v);
      this.headWorld.lerp(_v, 0.45);
    } else {
      // no skull-top joint: the head's centre sits ~9 cm up the head bone
      b.Head.getWorldQuaternion(_q);
      this.headWorld.add(_v.set(0, 0.09, 0.02).applyQuaternion(_q));
    }
    b.Spine2.getWorldPosition(this.chestWorld);
    b.Spine.getWorldPosition(this.bellyWorld);
    for (const s of SIDES) {
      const g = this.gloves[s];
      g.getWorldPosition(this.gloveWorld[s]);
      if (!this._first && dt > 0) {
        _v.subVectors(this.gloveWorld[s], this._glovePrev[s]).divideScalar(dt);
        this.gloveVel[s].lerp(_v, 0.6);
      }
      this._glovePrev[s].copy(this.gloveWorld[s]);
      // elbow point (for elbow strikes): the tip of the elbow sits a few cm
      // past the joint, along the upper arm
      b[s + 'ForeArm'].getWorldPosition(this.elbowWorld[s]);
      b[s + 'Arm'].getWorldPosition(_v2);
      _v2.subVectors(this.elbowWorld[s], _v2).normalize();
      this.elbowWorld[s].addScaledVector(_v2, 0.05);
      if (!this._first && dt > 0) {
        _v.subVectors(this.elbowWorld[s], this._elbowPrev[s]).divideScalar(dt);
        this.elbowVel[s].lerp(_v, 0.6);
      }
      this._elbowPrev[s].copy(this.elbowWorld[s]);
    }
    this._first = false;
  }

  /** Converts a world direction into this fighter's root space. */
  toRootDir(worldDir, out) {
    this.group.getWorldQuaternion(_q);
    return out.copy(worldDir).applyQuaternion(_q.invert());
  }

  toRootPos(worldPos, out) {
    return this.group.worldToLocal(out.copy(worldPos));
  }
}
