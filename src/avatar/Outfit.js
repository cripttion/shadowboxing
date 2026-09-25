// Turns the stock avatar into a boxer: crops the long jacket at the waist and
// swaps the suit materials for athletic gear. Done once on the shared source
// model, so both fighters reuse the same (cropped) geometry.
import * as THREE from 'three';

const WAIST_Y = 0.94; // bind-pose height below which jacket tails are removed

export function prepareAvatar(root) {
  root.updateMatrixWorld(true);
  root.traverse((o) => {
    if (!o.isMesh) return;
    const name = o.material?.name || '';
    if (/Outfit_Top/i.test(name)) cropBelow(o, WAIST_Y);
  });
}

function cropBelow(mesh, y) {
  const g = mesh.geometry;
  const pos = g.attributes.position;
  const idx = g.index;
  if (!idx) return;
  const keep = [];
  const src = idx.array;
  for (let i = 0; i < src.length; i += 3) {
    const a = src[i];
    const b = src[i + 1];
    const c = src[i + 2];
    if (Math.min(pos.getY(a), pos.getY(b), pos.getY(c)) < y) continue;
    keep.push(a, b, c);
  }
  g.setIndex(keep);
}

/**
 * Athletic kit. colors: { top, trunks, shoes, skin? }
 */
export function applyKit(mesh, kit) {
  const name = mesh.material.name || '';
  const old = mesh.material;
  if (/Outfit_Top/i.test(name)) {
    mesh.material = new THREE.MeshPhysicalMaterial({
      name,
      color: kit.top,
      roughness: 0.5,
      metalness: 0,
      clearcoat: 0.12,
      clearcoatRoughness: 0.6,
    });
  } else if (/Outfit_Bottom/i.test(name)) {
    // satin trunks (note: no `sheen` — it can emit NaNs that bloom smears across the frame)
    mesh.material = new THREE.MeshPhysicalMaterial({
      name,
      color: kit.trunks,
      roughness: 0.28,
      metalness: 0.05,
      clearcoat: 0.35,
      clearcoatRoughness: 0.3,
    });
  } else if (/Footwear/i.test(name)) {
    mesh.material = new THREE.MeshStandardMaterial({ name, color: kit.shoes, roughness: 0.45 });
  } else {
    mesh.material = old.clone();
    if (kit.skin && /Skin|Body/i.test(name)) mesh.material.color.set(kit.skin);
  }
  if (!kit.physical && mesh.material.isMeshPhysicalMaterial) {
    const m = mesh.material;
    mesh.material = new THREE.MeshStandardMaterial({ name, color: m.color, roughness: m.roughness, metalness: m.metalness });
  }
}
