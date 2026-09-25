// Selectable camera views. Each view returns a target position / look point /
// FOV; the rig eases between them (except FPP, which is locked to the head so
// it feels like your own eyes — slips and hit reactions move the view).
import * as THREE from 'three';

export const VIEWS = [
  { id: 'side', label: 'Side view' },
  { id: 'tpp', label: 'Third person' },
  { id: 'fpp', label: 'First person' },
  { id: 'front', label: 'Front view' },
  { id: 'top', label: 'Top view' },
  { id: 'cinematic', label: 'Cinematic' },
];

const _pos = new THREE.Vector3();
const _look = new THREE.Vector3();
const _v = new THREE.Vector3();
const _q = new THREE.Quaternion();

export class ViewRig {
  constructor(camera) {
    this.camera = camera;
    this.view = 'side';
    this.pos = new THREE.Vector3(3, 1.6, 0);
    this.look = new THREE.Vector3(0, 1.2, 0);
    this.fov = 45;
    this.near = 0.05;
    this.snap = true;
    this.cine = { shot: 0, t: 0, focusT: 0 };
    this._mid = 0;
  }

  set(view) {
    if (!VIEWS.some((v) => v.id === view)) view = 'side';
    this.view = view;
    this.snap = view === 'fpp';
    return view;
  }

  next() {
    const i = VIEWS.findIndex((v) => v.id === this.view);
    return this.set(VIEWS[(i + 1) % VIEWS.length].id);
  }

  label() {
    return VIEWS.find((v) => v.id === this.view)?.label || '';
  }

  /** Cinematic: cut to a close angle for a moment after a big hit. */
  focusHit() {
    if (this.view === 'cinematic') {
      this.cine.focusT = 1.4;
      this.snap = true;
    }
  }

  /**
   * @param ctx {player, opponent, zPlayer, zOpp, aspect, dt}
   * Writes the desired pose into this.pos / this.look / this.fov.
   */
  update(ctx) {
    const { player, opponent, zPlayer, zOpp, aspect, dt } = ctx;
    const portrait = aspect < 1;
    const baseFov = portrait ? 70 : 45;
    const mid = (zPlayer + zOpp) / 2;
    const gap = zPlayer - zOpp;
    this._mid += (mid - this._mid) * (1 - Math.exp(-dt * 5));
    player.rig.bones.Hips.getWorldPosition(_v);
    const follow = _v.x * 0.5;
    let fov = baseFov;
    let view = this.view;

    if (view === 'cinematic') {
      const c = this.cine;
      c.t += dt;
      if (c.focusT > 0) {
        c.focusT -= dt;
        view = 'close';
      } else if (c.t > 5.5) {
        c.t = 0;
        c.shot = (c.shot + 1) % 5;
        this.snap = true; // broadcast-style hard cut
      }
      if (view !== 'close') view = ['side', 'wide', 'tpp', 'low', 'front'][c.shot];
    }

    switch (view) {
      case 'tpp':
        // over the right shoulder, behind you, looking at the opponent
        _pos.set(0.55 + follow, 1.98, zPlayer + 1.75);
        _look.set(follow * 0.3 - 0.1, 1.32, zOpp + 0.15);
        break;
      case 'fpp': {
        // your eyes: head joint + a little up/forward along the head's facing
        const head = player.rig.bones.Head;
        head.getWorldPosition(_pos);
        head.getWorldQuaternion(_q);
        _v.set(0, 0.11, -0.01).applyQuaternion(_q); // eye height, not in front of the face
        _pos.add(_v);
        // gaze: mostly at the opponent, turned partly by your own head
        _look.copy(opponent.headWorld).lerp(opponent.chestWorld, 0.35);
        _v.set(0, 0, 1).applyQuaternion(_q).multiplyScalar(0.9).add(_pos);
        _look.lerp(_v, 0.3);
        fov = portrait ? 88 : 72;
        break;
      }
      case 'front':
        // behind the opponent's shoulder, facing you (like a mirror)
        _pos.set(-1.05, 1.85, zOpp - 1.3);
        _look.set(0, 1.4, zPlayer);
        break;
      case 'top':
        _pos.set(1.1, 5.4, this._mid + 0.7);
        _look.set(0, 1.0, this._mid);
        fov = baseFov + 8;
        break;
      case 'wide':
        _pos.set(2.6, 2.6, this._mid + 2.2);
        _look.set(0, 1.1, this._mid);
        break;
      case 'low':
        // low hero angle from the canvas, looking up at the exchange
        _pos.set(1.7, 0.55, this._mid + 0.9);
        _look.set(0, 1.5, this._mid);
        fov = baseFov + 5;
        break;
      case 'close':
        // tight on the opponent's reaction
        _pos.set(1.1, 1.65, zOpp + 0.9);
        _look.copy(opponent.headWorld);
        fov = baseFov - 5;
        break;
      default: {
        // side-on broadcast view: you on the left, opponent on the right;
        // stays inside the ropes and pulls back as the fighters separate
        const back = Math.min(3.05, 2.75 + Math.max(0, gap - 0.9) * 0.5);
        _pos.set(back, portrait ? 1.62 : 1.5, this._mid + 0.3);
        _look.set(0, 1.22, this._mid);
      }
    }

    this.near = 0.05;

    if (this.snap) {
      this.pos.copy(_pos);
      this.look.copy(_look);
      this.fov = fov;
      this.snap = this.view === 'fpp'; // FPP stays glued to the head
    } else {
      const k = 1 - Math.exp(-dt * 6);
      this.pos.lerp(_pos, k);
      this.look.lerp(_look, k);
      this.fov += (fov - this.fov) * k;
    }
  }
}
