// BlazePose landmark indices (33-point topology) and the packed buffer layout
// shared by the worker, the main-thread fallback and the filters.

export const LM = {
  NOSE: 0,
  L_EYE: 2,
  R_EYE: 5,
  L_EAR: 7,
  R_EAR: 8,
  L_SHOULDER: 11,
  R_SHOULDER: 12,
  L_ELBOW: 13,
  R_ELBOW: 14,
  L_WRIST: 15,
  R_WRIST: 16,
  L_PINKY: 17,
  R_PINKY: 18,
  L_INDEX: 19,
  R_INDEX: 20,
  L_HIP: 23,
  R_HIP: 24,
  L_KNEE: 25,
  R_KNEE: 26,
  L_ANKLE: 27,
  R_ANKLE: 28,
};

export const NUM_LANDMARKS = 33;
// Per landmark: x, y, z, visibility
export const STRIDE = 4;

// Bones drawn by the camera preview overlay
export const SKELETON_EDGES = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
  [15, 19], [16, 20], [0, 7], [0, 8],
];

/** Pack MediaPipe landmark objects into a flat Float32Array (x,y,z,vis). */
export function packLandmarks(list, out) {
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const p = list[i];
    const o = i * STRIDE;
    out[o] = p.x;
    out[o + 1] = p.y;
    out[o + 2] = p.z;
    out[o + 3] = p.visibility ?? 1;
  }
  return out;
}
