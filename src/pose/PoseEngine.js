// Thin wrapper around MediaPipe PoseLandmarker. Runs identically inside the
// pose worker (preferred) and on the main thread (fallback).
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { NUM_LANDMARKS, STRIDE, packLandmarks } from './landmarks.js';

export class PoseEngine {
  constructor() {
    this.landmarker = null;
    this.delegate = null;
    this.lastTs = 0;
  }

  /**
   * @param {{wasmBase:string, modelUrl:string, delegate?:'GPU'|'CPU', useModule?:boolean}} opts
   */
  async init({ wasmBase, modelUrl, delegate = 'GPU', useModule = true }) {
    const fileset = await FilesetResolver.forVisionTasks(wasmBase, useModule);
    const create = async (d) => {
      if (useModule) {
        // MediaPipe deletes the global factory after instantiating, and ES
        // modules only evaluate once — re-publish it so another engine (CPU
        // fallback, or the GPU/CPU auto-benchmark) can start too.
        const mod = await import(/* @vite-ignore */ fileset.wasmLoaderPath);
        globalThis.ModuleFactory = mod.default;
      }
      return PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelUrl, delegate: d },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputSegmentationMasks: false,
      });
    };
    try {
      this.landmarker = await create(delegate);
      this.delegate = delegate;
    } catch (err) {
      if (delegate !== 'GPU') throw err;
      // Older / blocklisted GPUs: fall back to the XNNPACK CPU path.
      console.warn('[pose] GPU delegate failed, using CPU', err);
      this.landmarker = await create('CPU');
      this.delegate = 'CPU';
    }
    return this.delegate;
  }

  /**
   * Run inference on one frame.
   * @returns {{world:Float32Array, norm:Float32Array}|null}
   */
  detect(image, tsMs) {
    // VIDEO mode requires strictly increasing timestamps
    const ts = Math.max(Math.round(tsMs), this.lastTs + 1);
    this.lastTs = ts;
    const res = this.landmarker.detectForVideo(image, ts);
    if (!res.landmarks || res.landmarks.length === 0) return null;
    const world = packLandmarks(res.worldLandmarks[0], new Float32Array(NUM_LANDMARKS * STRIDE));
    const norm = packLandmarks(res.landmarks[0], new Float32Array(NUM_LANDMARKS * STRIDE));
    return { world, norm };
  }

  close() {
    this.landmarker?.close();
    this.landmarker = null;
  }
}
