// Physical hit detection: gloves are swept spheres tested against the
// defender's gloves (block), head and torso. Swept tests (segment vs sphere)
// mean a fast punch can't tunnel through a head between two frames, even at
// low frame rates.
import * as THREE from 'three';

const GLOVE_R = 0.075;
const ELBOW_R = 0.09; // elbow + the forearm behind it
const HEAD_R = 0.13;
const BODY_R = 0.16;

const _ab = new THREE.Vector3();
const _ac = new THREE.Vector3();
const _p = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _rel = new THREE.Vector3();

/** Returns the closest point parameter t∈[0,1] if the segment a→b touches the sphere. */
function sweep(a, b, c, r) {
  _ab.subVectors(b, a);
  _ac.subVectors(c, a);
  const l2 = _ab.lengthSq();
  const t = l2 > 1e-9 ? Math.min(1, Math.max(0, _ac.dot(_ab) / l2)) : 0;
  _p.copy(a).addScaledVector(_ab, t);
  return _p.distanceToSquared(c) <= r * r ? t : -1;
}

export class Combat {
  constructor() {
    this.prev = new Map(); // fighter -> {Left, Right} previous glove positions
    this.armed = new Map(); // fighter -> {Left, Right} can this glove land a hit
  }

  _state(f) {
    if (!this.prev.has(f)) {
      this.prev.set(f, {
        glove: { Left: f.gloveWorld.Left.clone(), Right: f.gloveWorld.Right.clone() },
        elbow: { Left: f.elbowWorld.Left.clone(), Right: f.elbowWorld.Right.clone() },
      });
      this.armed.set(f, { Left: true, Right: true });
    }
    return { prev: this.prev.get(f), armed: this.armed.get(f) };
  }

  /**
   * Test attacker's gloves against defender.
   * @param isStriking (side, speed) => boolean — whether that glove is "live" (punch intent)
   * @param guarding boolean | (side) => boolean — can that defender glove block
   *        (a glove held by the face covers it; a fast swat parries)
   * @returns array of hit events
   */
  check(attacker, defender, isStriking, guarding = true) {
    const canBlock = typeof guarding === 'function' ? guarding : () => guarding;
    const { prev, armed } = this._state(attacker);
    const events = [];
    for (const s of ['Left', 'Right']) {
      // isStriking returns false, true/'glove', or 'elbow' (the striking point)
      const strike = isStriking(s, attacker.gloveVel[s].length());
      const elbow = strike === 'elbow';
      const a = elbow ? prev.elbow[s] : prev.glove[s];
      const b = elbow ? attacker.elbowWorld[s] : attacker.gloveWorld[s];
      const v = elbow ? attacker.elbowVel[s] : attacker.gloveVel[s];
      const R = elbow ? ELBOW_R : GLOVE_R;
      const speed = v.length();
      const live = !!strike;
      // re-arm once the glove has come back / slowed down
      if (!live && speed < 1.2) armed[s] = true;
      if (live && armed[s]) {
        let hit = null;
        // gloves first: a guard in the way blocks the punch
        for (const ds of ['Left', 'Right']) {
          if (!canBlock(ds)) continue;
          if (sweep(a, b, defender.gloveWorld[ds], R + GLOVE_R) >= 0) {
            hit = { zone: 'block', defSide: ds, point: _p.clone().lerp(defender.gloveWorld[ds], 0.5) };
            break;
          }
        }
        if (!hit && sweep(a, b, defender.headWorld, HEAD_R + R) >= 0) {
          hit = { zone: 'head', point: _p.clone().lerp(defender.headWorld, 0.5) };
        }
        if (!hit) {
          for (const c of [defender.chestWorld, defender.bellyWorld]) {
            if (sweep(a, b, c, BODY_R + R) >= 0) {
              hit = { zone: 'body', point: _p.clone().lerp(c, 0.4) };
              break;
            }
          }
        }
        if (hit) {
          armed[s] = false;
          _dir.copy(v).normalize();
          hit.side = s;
          hit.speed = speed;
          hit.dir = _dir.clone();
          hit.kind = classify(attacker, s, v);
          hit.striker = elbow ? 'elbow' : 'glove';
          events.push(hit);
        }
      }
      prev.glove[s].copy(attacker.gloveWorld[s]);
      prev.elbow[s].copy(attacker.elbowWorld[s]);
    }
    return events;
  }
}

/** Label a punch from the glove velocity in the attacker's own frame. */
function classify(fighter, side, vWorld) {
  fighter.toRootDir(vWorld, _rel);
  const fwd = _rel.z;
  const up = _rel.y;
  const lat = Math.abs(_rel.x);
  if (up > fwd * 0.9 && up > lat) return 'UPPERCUT';
  if (lat > fwd * 0.85) return 'HOOK';
  const lead = fighter.stanceSign > 0 ? 'Left' : 'Right';
  return side === lead ? 'JAB' : 'CROSS';
}
