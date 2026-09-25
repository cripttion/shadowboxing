// Skeleton solver shared by every driver (camera retargeting, AI, keyboard).
//
// All maths happens in "root space": the character's own frame where the
// model faces +Z, up is +Y and the character's LEFT is +X. Drivers only ever
// produce directions / targets in that frame, so the same code animates the
// player (turned to face the opponent) and the opponent.
import * as THREE from 'three';

const _q0 = new THREE.Quaternion();
const _q1 = new THREE.Quaternion();
const _q2 = new THREE.Quaternion();
const _v0 = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _m0 = new THREE.Matrix4();
const _m1 = new THREE.Matrix4();
const _identity = new THREE.Quaternion();

export const FORWARD = new THREE.Vector3(0, 0, 1);
export const UP = new THREE.Vector3(0, 1, 0);
export const LEFT = new THREE.Vector3(1, 0, 0);

/** Rotation matrix whose columns are an orthonormal frame built from (primary, secondary). */
function frameMatrix(primary, secondary, out) {
  const x = _v0.copy(primary).normalize();
  const y = _v1.copy(secondary).addScaledVector(x, -secondary.dot(x));
  if (y.lengthSq() < 1e-8) y.set(0, 1, 0).addScaledVector(x, -x.y);
  y.normalize();
  const z = _v2.crossVectors(x, y);
  return out.makeBasis(x, y, z);
}

/** Quaternion that rotates frame(restA,restB) onto frame(tA,tB). */
export function frameRotation(restA, restB, tA, tB, out) {
  frameMatrix(restA, restB, _m0);
  // copy because frameMatrix reuses temp vectors
  const rest = _m1.copy(_m0);
  frameMatrix(tA, tB, _m0);
  rest.transpose(); // inverse of an orthonormal basis
  _m0.multiply(rest);
  return out.setFromRotationMatrix(_m0);
}

export class Rig {
  constructor(root) {
    this.root = root;
    this.bones = {};
    root.traverse((o) => {
      if (o.isBone) {
        const n = o.name.replace(/^mixamorig[:_]?/, '');
        if (!this.bones[n]) this.bones[n] = o;
      }
    });
    root.updateMatrixWorld(true);
    const rootInv = new THREE.Matrix4().copy(root.matrixWorld).invert();
    const rootQInv = root.getWorldQuaternion(new THREE.Quaternion()).invert();

    this.rest = new Map();
    for (const b of Object.values(this.bones)) {
      const pos = new THREE.Vector3().setFromMatrixPosition(b.matrixWorld).applyMatrix4(rootInv);
      const worldQ = b.getWorldQuaternion(new THREE.Quaternion()).premultiply(rootQInv);
      this.rest.set(b, { local: b.quaternion.clone(), localPos: b.position.clone(), worldQ, pos });
    }

    this.rootQ = new THREE.Quaternion();
    this.rootQInv = new THREE.Quaternion();

    // Cached segment lengths
    const len = (a, b) => this.restPos(a).distanceTo(this.restPos(b));
    this.upperArmLen = len('LeftArm', 'LeftForeArm');
    this.foreArmLen = len('LeftForeArm', 'LeftHand');
    this.thighLen = len('LeftUpLeg', 'LeftLeg');
    this.shinLen = len('LeftLeg', 'LeftFoot');
    this.shoulderWidth = len('LeftArm', 'RightArm');
    this.hipsRest = this.restPos('Hips').clone();
    this.headRest = this.restPos('Head').clone();

    // Rest frames used by frame-matching solvers
    this.restTorsoUp = new THREE.Vector3().subVectors(this.restPos('Neck'), this.restPos('Hips')).normalize();
    this.restRight = new THREE.Vector3().subVectors(this.restPos('RightArm'), this.restPos('LeftArm')).normalize();
    this.restHipRight = new THREE.Vector3().subVectors(this.restPos('RightUpLeg'), this.restPos('LeftUpLeg')).normalize();

    // Arm bend-plane reference: an elbow flexing in the bind pose moves the
    // forearm toward the character's front.
    this.restArm = {};
    for (const s of ['Left', 'Right']) {
      const dir = new THREE.Vector3().subVectors(this.restPos(s + 'ForeArm'), this.restPos(s + 'Arm')).normalize();
      const normal = new THREE.Vector3().crossVectors(dir, FORWARD).normalize();
      this.restArm[s] = { dir, normal };
    }
    this._armNormal = {
      Left: new THREE.Vector3(0, -1, 0),
      Right: new THREE.Vector3(0, 1, 0),
    };
  }

  restPos(name) {
    return this.rest.get(this.bones[name]).pos;
  }

  /** Must be called once per frame before solving. */
  begin() {
    this.root.updateWorldMatrix(true, false);
    this.root.getWorldQuaternion(this.rootQ);
    this.rootQInv.copy(this.rootQ).invert();
  }

  resetToRest() {
    for (const [b, r] of this.rest) {
      b.quaternion.copy(r.local);
      b.position.copy(r.localPos);
    }
  }

  /** Bone orientation in root space. */
  spaceQ(obj, out) {
    obj.getWorldQuaternion(out);
    return out.premultiply(this.rootQInv);
  }

  /** Bone position in root space. */
  spacePos(obj, out) {
    obj.updateWorldMatrix(true, false);
    out.setFromMatrixPosition(obj.matrixWorld);
    return this.root.worldToLocal(out);
  }

  pos(name, out) {
    return this.spacePos(this.bones[name], out);
  }

  /** Set a bone's orientation given in root space. */
  setSpaceQ(bone, q) {
    this.spaceQ(bone.parent, _q0);
    bone.quaternion.copy(_q0.invert().multiply(q));
  }

  /** Move a bone (typically Hips) to a root-space position. */
  setSpacePos(bone, p) {
    _v3.copy(p);
    this.root.localToWorld(_v3);
    bone.parent.updateWorldMatrix(true, false);
    bone.parent.worldToLocal(_v3);
    bone.position.copy(_v3);
  }

  /** Orient bone = rotation(restFrame → targetFrame) applied to its rest orientation. */
  setFrame(name, rotation, weight = 1) {
    const b = this.bones[name];
    const r = this.rest.get(b);
    _q1.copy(_identity).slerp(rotation, weight);
    _q1.multiply(r.worldQ);
    this.setSpaceQ(b, _q1);
  }

  /**
   * Swing a bone (from its rest local orientation) so the segment towards
   * `childName` points along `dir` (root space).
   */
  aim(name, childName, dir, weight = 1) {
    const b = this.bones[name];
    const c = this.bones[childName];
    const r = this.rest.get(b);
    b.quaternion.copy(r.local);
    this.spaceQ(b, _q1);
    _v0.copy(this.rest.get(c).localPos).normalize().applyQuaternion(_q1);
    _v1.copy(dir).normalize();
    _q2.setFromUnitVectors(_v0, _v1);
    if (weight < 1) _q2.slerp(_identity, 1 - weight);
    _q1.premultiply(_q2);
    this.setSpaceQ(b, _q1);
  }

  /**
   * Arm solve with correct twist: the upper arm is oriented from BOTH its
   * direction and the elbow bend plane, so elbows point where the real elbow
   * points instead of spinning around the bone axis.
   */
  solveArm(side, upperDir, foreDir) {
    const upper = this.bones[side + 'Arm'];
    const ra = this.restArm[side];
    const n = _v2.crossVectors(upperDir, foreDir);
    const len = n.length();
    const prevN = this._armNormal[side];
    if (len > 0.25) {
      prevN.copy(n).divideScalar(len);
    } else if (len > 0.02) {
      // nearly straight: trust the measurement proportionally to its confidence
      prevN.lerp(n.divideScalar(len), (len - 0.02) / 0.23).normalize();
    }
    frameRotation(ra.dir, ra.normal, upperDir, prevN, _q2);
    _q2.multiply(this.rest.get(upper).worldQ);
    this.setSpaceQ(upper, _q2);
    this.aim(side + 'ForeArm', side + 'Hand', foreDir);
    const hand = this.bones[side + 'Hand'];
    hand.quaternion.copy(this.rest.get(hand).local);
  }

  /**
   * Analytic two-bone IK: returns upper & lower directions reaching `target`
   * from `origin`, bending toward `pole`.
   */
  static twoBone(origin, target, l1, l2, pole, outUpper, outLower) {
    const toT = _v0.subVectors(target, origin);
    let d = toT.length();
    const u = toT.divideScalar(d || 1);
    d = Math.min(Math.max(d, Math.abs(l1 - l2) + 1e-3), (l1 + l2) * 0.9999);
    const a = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
    const h = Math.sqrt(Math.max(0, l1 * l1 - a * a));
    const v = _v1.copy(pole).addScaledVector(u, -pole.dot(u));
    if (v.lengthSq() < 1e-8) v.set(0, -1, 0).addScaledVector(u, -u.y);
    v.normalize();
    // mid joint relative to origin
    const mid = _v2.copy(u).multiplyScalar(a).addScaledVector(v, h);
    outUpper.copy(mid).normalize();
    outLower.copy(u).multiplyScalar(d).sub(mid).normalize();
  }
}
