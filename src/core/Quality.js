// Device-tier detection + tier presets. Everything that costs GPU time is
// driven from here so a low-end phone and a gaming PC run the same code.

export const TIERS = {
  low: { maxDpr: 1, minDpr: 0.55, shadows: false, shadowSize: 0, bloom: false, crowd: 220, physicalGloves: false, dust: 120, poseModel: 'lite' },
  medium: { maxDpr: 1.5, minDpr: 0.7, shadows: true, shadowSize: 1024, bloom: false, crowd: 520, physicalGloves: true, dust: 300, poseModel: 'lite' },
  high: { maxDpr: 2, minDpr: 0.85, shadows: true, shadowSize: 2048, bloom: true, crowd: 900, physicalGloves: true, dust: 600, poseModel: 'full' },
};

export function detectTier() {
  const saved = safeGet('sb.quality');
  if (saved && saved !== 'auto' && TIERS[saved]) return saved;
  let score = 0;
  const cores = navigator.hardwareConcurrency || 4;
  const mem = navigator.deviceMemory || 4;
  const mobile = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
  score += cores >= 8 ? 2 : cores >= 6 ? 1 : 0;
  score += mem >= 8 ? 2 : mem >= 4 ? 1 : 0;
  if (mobile) score -= 2;
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2');
    const ext = gl && gl.getExtension('WEBGL_debug_renderer_info');
    const r = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : '';
    if (/Apple M\d|RTX|Radeon RX|GeForce GTX 1[0-9]{3}|Arc/i.test(r)) score += 2;
    if (/Intel.*(HD|UHD)|Mali-[GT]\d{1,2}\b|Adreno \(TM\) [3-5]\d\d|PowerVR|SwiftShader|llvmpipe/i.test(r)) score -= 2;
    gl?.getExtension('WEBGL_lose_context')?.loseContext();
  } catch {
    score -= 1;
  }
  if (score >= 4) return 'high';
  if (score >= 1) return 'medium';
  return 'low';
}

export function safeGet(k) {
  try {
    return localStorage.getItem(k);
  } catch {
    return null;
  }
}

export function safeSet(k, v) {
  try {
    localStorage.setItem(k, v);
  } catch {
    /* private mode */
  }
}
