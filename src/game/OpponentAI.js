// Opponent brain: decides when to attack, which combos to throw, when to
// block or slip, and how to react to getting hit. Drives a ProceduralBoxer.
import * as THREE from 'three';

export const DIFFICULTY = {
  easy: { cooldown: [1.9, 3.3], tell: 0.6, block: 0.2, slip: 0.05, react: 0.3, dmg: 0.65, combos: 0 },
  normal: { cooldown: [1.2, 2.3], tell: 0.45, block: 0.4, slip: 0.15, react: 0.22, dmg: 1.0, combos: 1 },
  hard: { cooldown: [0.65, 1.5], tell: 0.32, block: 0.58, slip: 0.25, react: 0.15, dmg: 1.3, combos: 2 },
};

const COMBOS = [
  // tier 0
  [['jab'], ['cross'], ['jab', 'cross'], ['hook'], ['bodyhook']],
  // tier 1
  [['jab', 'jab', 'cross'], ['jab', 'cross', 'hook'], ['body', 'hook'], ['cross', 'hook'], ['uppercut'], ['jab', 'overhand'], ['lupper', 'cross'], ['hook', 'relbow']],
  // tier 2
  [['jab', 'cross', 'hook', 'cross'], ['hook', 'uppercut'], ['jab', 'body', 'hook'], ['cross', 'rhook'], ['jab', 'uppercut', 'hook'], ['bodyhook', 'hook', 'cross'], ['jab', 'cross', 'overhand'], ['lupper', 'rhook', 'bodyhook'], ['jab', 'cross', 'lelbow'], ['bodyhook', 'relbow']],
];

const _v = new THREE.Vector3();
const _d = new THREE.Vector3();

export class OpponentAI {
  constructor(boxer, self, target, difficulty = 'normal') {
    this.boxer = boxer;
    this.self = self; // Fighter this AI drives
    this.target = target; // Fighter it fights
    this.setDifficulty(difficulty);
    this.cooldown = 1.5;
    this.blockTimer = 0;
    this.slipTimer = 0;
    this.reactTimer = 0;
    this.stun = 0;
    this.recentHits = [];
    this.threatSeen = false;
    this.enabled = true;
    this.aggression = 1;
  }

  setDifficulty(d) {
    this.diffName = d;
    this.diff = DIFFICULTY[d] || DIFFICULTY.normal;
  }

  /** Root-space aim point for a punch at the target's head (or body). */
  _aim(out, body) {
    const t = this.target;
    _v.copy(body ? t.bellyWorld : t.headWorld);
    if (body) _v.lerp(t.chestWorld, 0.35);
    this.self.toRootPos(_v, out);
    // aim a little *through* the target so the punch has intent
    _d.copy(out);
    _d.y = 0;
    const len = _d.length() || 1;
    out.addScaledVector(_d, 0.1 / len);
    return out;
  }

  onHit(power) {
    const now = performance.now();
    this.recentHits.push(now);
    this.recentHits = this.recentHits.filter((t) => now - t < 1600);
    const wasWindingUp = this.boxer.action && this.boxer.action.phase === 'windup';
    this.boxer.cancel();
    this.blockTarget = 0;
    this.cooldown = Math.max(this.cooldown, 0.5 + Math.random() * 0.4);
    if (this.recentHits.length >= 3 || power > 1.35) {
      this.stun = 1.1;
      this.recentHits.length = 0;
    }
    return wasWindingUp;
  }

  update(dt) {
    const b = this.boxer;
    const d = this.diff;
    if (!this.enabled) {
      b.blockTarget = 0;
      return;
    }

    this.stun = Math.max(0, this.stun - dt);
    b.idleAmp = this.stun > 0 ? 0.35 : 1;
    b.guard.Left.y = b.guard.Right.y = b.headY - (this.stun > 0 ? 0.28 : 0.1);

    // --- Defence: read incoming gloves --------------------------------------
    let threat = false;
    for (const s of ['Left', 'Right']) {
      const g = this.target.gloveWorld[s];
      const v = this.target.gloveVel[s];
      _d.subVectors(this.self.headWorld, g);
      const dist = _d.length();
      if (dist < 0.95 && v.dot(_d) / (dist || 1) > 1.8) threat = true;
    }
    if (threat && !this.threatSeen) {
      this.threatSeen = true;
      this.reactTimer = d.react * (0.8 + Math.random() * 0.5);
      this._pendingDefence = Math.random();
    }
    if (!threat) this.threatSeen = false;
    if (this.reactTimer > 0) {
      this.reactTimer -= dt;
      if (this.reactTimer <= 0 && this.stun <= 0 && !(b.action && b.action.phase === 'strike')) {
        const r = this._pendingDefence;
        if (r < d.slip) {
          this.slipTimer = 0.42;
          b.slip = (Math.random() < 0.5 ? -1 : 1) * 0.2;
        } else if (r < d.slip + d.block) {
          this.blockTimer = 0.55;
          b.cancel();
        }
      }
    }
    this.blockTimer = Math.max(0, this.blockTimer - dt);
    this.slipTimer = Math.max(0, this.slipTimer - dt);
    b.blockTarget = this.blockTimer > 0 ? 1 : 0;
    if (this.slipTimer <= 0) b.slip *= Math.exp(-dt * 8);

    // --- Offence --------------------------------------------------------------
    this.cooldown -= dt * this.aggression;
    if (this.cooldown <= 0 && !b.busy && this.stun <= 0 && this.blockTimer <= 0) {
      const tier = Math.min(COMBOS.length - 1, Math.floor(Math.random() * (d.combos + 1.4)));
      const list = COMBOS[tier];
      const combo = list[(Math.random() * list.length) | 0];
      combo.forEach((name, i) => {
        const body = name === 'body' || name === 'bodyhook';
        b.punch(name, (out) => this._aim(out, body), { windup: i === 0 ? d.tell * (0.85 + Math.random() * 0.3) : undefined });
      });
      const [a, z] = d.cooldown;
      this.cooldown = a + Math.random() * (z - a) + combo.length * 0.25;
    }
  }
}
