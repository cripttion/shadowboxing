// Dev-only regression test for camera punch detection. Load the game, then in
// the console: await import('/dev/punch-test.js').then(m => m.run())
// Feeds synthetic MediaPipe world landmarks (30 fps) through the real
// filter → CameraDriver → Fighter → Combat path and reports what registered.
const N = 33;
const GUARD = {
  0: [0, -0.65, -0.1], 7: [0.07, -0.66, 0], 8: [-0.07, -0.66, 0],
  11: [0.18, -0.48, 0], 12: [-0.18, -0.48, 0],
  13: [0.22, -0.25, -0.1], 14: [-0.22, -0.25, -0.1],
  15: [0.1, -0.5, -0.22], 16: [-0.1, -0.5, -0.22],
  23: [0.1, 0, 0], 24: [-0.1, 0, 0],
};
// torso turns with rear-hand punches (right shoulder forward), guard hand stays at the chin
const TURN_R = { 11: [0.15, -0.48, 0.1], 12: [-0.15, -0.48, -0.1], 15: [0.07, -0.52, -0.18] };
const TURN_L = { 11: [0.15, -0.48, -0.1], 12: [-0.15, -0.48, 0.1], 16: [-0.07, -0.52, -0.18] };
export const MOVES = {
  cross: { expect: 'Right', pose: { ...TURN_R, 14: [-0.09, -0.49, -0.36], 16: [-0.02, -0.48, -0.62] } },
  jab: { expect: 'Left', pose: { ...TURN_L, 13: [0.09, -0.49, -0.36], 15: [0.02, -0.48, -0.62] } },
  hook: { expect: 'Left', from: { 13: [0.3, -0.42, -0.02], 15: [0.3, -0.5, -0.12] }, pose: { ...TURN_L, 13: [0.36, -0.5, -0.12], 15: [-0.06, -0.52, -0.32] } },
  uppercut: { expect: 'Right', from: { 16: [-0.12, -0.28, -0.2] }, pose: { ...TURN_R, 14: [-0.18, -0.27, -0.16], 16: [-0.05, -0.62, -0.44] } },
  body: { expect: 'Right', pose: { ...TURN_R, 14: [-0.14, -0.3, -0.26], 16: [-0.05, -0.16, -0.52] } },
  // lead hook thrown low, across into the liver
  bodyHook: { expect: 'Left', from: { 13: [0.3, -0.25, 0.0], 15: [0.3, -0.22, -0.1] }, pose: { ...TURN_L, 13: [0.34, -0.26, -0.12], 15: [-0.04, -0.2, -0.32] } },
  // rear hand loops up over the guard and comes down onto the head
  overhand: { expect: 'Right', from: { 14: [-0.3, -0.62, 0.05], 16: [-0.2, -0.72, -0.05] }, pose: { ...TURN_R, 14: [-0.12, -0.56, -0.3], 16: [-0.02, -0.5, -0.58] } },
  // must NOT register
  twitch: { expect: null, pose: { 15: [0.14, -0.53, -0.28], 16: [-0.06, -0.47, -0.27] } },
  shuffle: { expect: null, pose: { 11: [0.2, -0.47, 0.03], 12: [-0.16, -0.49, -0.03], 15: [0.13, -0.5, -0.2], 16: [-0.07, -0.51, -0.24] } },
  dropHands: { expect: null, pose: { 13: [0.24, -0.2, 0], 14: [-0.24, -0.2, 0], 15: [0.24, 0.02, -0.02], 16: [-0.24, 0.02, -0.02] } },
  torsoTurnOnly: { expect: null, pose: { ...TURN_R, 16: [-0.08, -0.5, -0.28] } },
  // elbows: fist tucked by the ear, raised elbow driven across into the target
  rearElbow: { expect: 'Right', from: { 14: [-0.38, -0.47, 0.05], 16: [-0.14, -0.62, -0.02] }, pose: { ...TURN_R, 14: [-0.02, -0.47, -0.3], 16: [-0.12, -0.6, -0.08] } },
  leadElbow: { expect: 'Left', from: { 13: [0.38, -0.47, 0.05], 15: [0.14, -0.62, -0.02] }, pose: { ...TURN_L, 13: [0.02, -0.47, -0.3], 15: [0.12, -0.6, -0.08] } },
  // raising both elbows slowly (a stretch): not a strike
  slowElbowsUp: { expect: null, go: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1], pose: { 13: [0.36, -0.46, -0.02], 14: [-0.36, -0.46, -0.02], 15: [0.12, -0.62, -0.05], 16: [-0.12, -0.62, -0.05] } },
  // quick elbow flare OUTWARD (away from the opponent line): not a strike
  elbowFlare: { expect: null, pose: { 14: [-0.42, -0.45, 0.06], 16: [-0.14, -0.6, -0.05] } },
  // arms spread wide, then brought back in to guard: not a hook
  spreadReturn: { expect: null, from: { 13: [0.4, -0.45, 0], 14: [-0.4, -0.45, 0], 15: [0.45, -0.5, -0.1], 16: [-0.45, -0.5, -0.1] }, pose: {} },
  // lazy arm extension over ~0.6 s: not a punch
  slowPush: { expect: null, go: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1], pose: { 14: [-0.1, -0.49, -0.36], 16: [-0.04, -0.48, -0.6] } },
  // quick but only half extended: not a punch
  halfPunch: { expect: null, pose: { 14: [-0.16, -0.4, -0.2], 16: [-0.08, -0.48, -0.36] } },
};

export async function run({ trackMode = 'full' } = {}) {
  const g = window.__game;
  g.trackMode = trackMode;
  g.poseService = { aspect: 4 / 3, stats: {}, video: null, stop() {} };
  g.mode = 'camera';
  g._prepareMatch();
  g.phase = 'fight';
  g.oppAI.update = (dt) => {
    g.oppAI.cooldown = 5;
    g.oppAI.stun = Math.max(0, g.oppAI.stun - dt);
  };
  document.querySelector('#menu').classList.add('hidden');
  g.hud.show(true);
  let ts = performance.now() + 10000;
  const mix = (a, b, t) => {
    const o = {};
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      const p = a[k] || GUARD[k];
      const q = b[k] || GUARD[k];
      o[k] = p.map((v, i) => v + (q[i] - v) * t);
    }
    return o;
  };
  const frame = (over) => {
    const w = new Float32Array(N * 4);
    const n = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) {
      const p = over[i] || GUARD[i] || [0, 0.3, 0];
      w.set([p[0], p[1], p[2], 0.99], i * 4);
      n.set([0.5 + (p[0] * 0.5) / (4 / 3), 0.4 + p[1] * 0.5, 0, 0.99], i * 4);
    }
    ts += 33.3;
    return { world: w, norm: n, ts };
  };
  const step = (over = {}) => {
    g.filter.push(frame(over));
    g._update(1 / 30, 1 / 30);
  };
  for (let i = 0; i < 30; i++) step();
  g.cameraDriver.calibrate(g.filter.sample(performance.now()), 4 / 3);
  g.playerAnchorZ = g.zPlayer;
  for (let i = 0; i < 15; i++) step();

  const results = [];
  let pass = 0;
  for (const [name, mv] of Object.entries(MOVES)) {
    // independent trials: fighters back at normal range, AI not stunned
    g.oppAI.stun = 0;
    g._resetPositions();
    g.playerAnchorZ = g.zPlayer;
    const from = mv.from || {};
    for (const t of [0.34, 0.67, 1, 1, 1]) step(mix({}, from, t)); // load (slow)
    const ids = { Left: g.cameraDriver.punch.Left.id, Right: g.cameraDriver.punch.Right.id };
    const hp = g.hp.ai;
    const kinds = {};
    for (const t of mv.go || [0.4, 0.8, 1, 1, 1]) {
      step(mix(from, mv.pose, t));
      for (const s of ['Left', 'Right']) if (g.cameraDriver.punch[s].active) kinds[s] = g.cameraDriver.punch[s].kind;
    }
    for (const t of [0.25, 0.5, 0.75, 1, 1, 1, 1, 1]) step(mix(mv.pose, {}, t));
    for (let i = 0; i < 20; i++) step();
    const fired = ['Left', 'Right'].filter((s) => g.cameraDriver.punch[s].id !== ids[s]);
    const dmg = hp - g.hp.ai;
    g.hp.ai = 100;
    const ok = mv.expect ? fired.length === 1 && fired[0] === mv.expect && dmg > 0 : fired.length === 0 && dmg === 0;
    if (ok) pass++;
    results.push(`${ok ? 'PASS' : 'FAIL'} ${name}: fired=[${fired.map((s) => s + ':' + (kinds[s] || '?')).join(',')}] dmg=${dmg}`);
  }
  // 1-2 combo: jab then cross straight after — must be two distinct punches
  {
    const ids = { Left: g.cameraDriver.punch.Left.id, Right: g.cameraDriver.punch.Right.id };
    const J = MOVES.jab.pose;
    const C = MOVES.cross.pose;
    for (const t of [0.4, 0.8, 1, 1]) step(mix({}, J, t));
    for (const t of [0.5, 1]) step(mix(J, {}, t));
    for (const t of [0.4, 0.8, 1, 1]) step(mix({}, C, t));
    for (const t of [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1]) step(mix(C, {}, t));
    const L = g.cameraDriver.punch.Left.id - ids.Left;
    const R = g.cameraDriver.punch.Right.id - ids.Right;
    const ok = L === 1 && R === 1;
    if (ok) pass++;
    results.push(`${ok ? 'PASS' : 'FAIL'} oneTwo combo: left punches=${L} right punches=${R}`);
  }
  results.push(`${pass}/${Object.keys(MOVES).length + 1} passed`);
  return results;
}
