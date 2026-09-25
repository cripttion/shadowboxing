// Procedural boxing glove (2 draw calls). Local frame: +Z = knuckles
// direction, +X = thumb side, -Z = cuff/wrist.
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

const cache = new Map();

function buildGeometries() {
  // Fist body: an ellipsoid sculpted flatter on the palm and fuller at the knuckles
  const fist = new THREE.SphereGeometry(1, 36, 24);
  const p = fist.attributes.position;
  for (let i = 0; i < p.count; i++) {
    let x = p.getX(i);
    let y = p.getY(i);
    let z = p.getZ(i);
    const front = Math.max(0, z);
    x *= 0.066 * (1 + 0.1 * front);
    y *= (y < 0 ? 0.047 : 0.064) * (1 + 0.12 * front);
    z *= z < 0 ? 0.075 : 0.092;
    p.setXYZ(i, x, y, z + 0.022);
  }
  const thumb = new THREE.CapsuleGeometry(0.023, 0.07, 6, 14);
  thumb.rotateX(Math.PI / 2);
  thumb.rotateY(-0.25);
  thumb.translate(0.052, -0.008, 0.012);
  const leather = mergeGeometries([fist.toNonIndexed(), thumb.toNonIndexed()]);
  leather.computeVertexNormals();

  const cuff = new THREE.CylinderGeometry(0.056, 0.05, 0.1, 28, 1, true);
  cuff.rotateX(Math.PI / 2);
  cuff.translate(0, -0.004, -0.085);
  const band = new THREE.TorusGeometry(0.054, 0.008, 8, 28);
  band.translate(0, -0.004, -0.058);
  const trim = mergeGeometries([cuff.toNonIndexed(), band.toNonIndexed()]);
  trim.computeVertexNormals();
  return { leather, trim };
}

export function createGlove(color, trimColor = 0xd9d9d9) {
  if (!cache.has('geo')) cache.set('geo', buildGeometries());
  const { leather, trim } = cache.get('geo');
  const g = new THREE.Group();
  const leatherMat = new THREE.MeshPhysicalMaterial({
    color,
    roughness: 0.38,
    metalness: 0,
    clearcoat: 0.7,
    clearcoatRoughness: 0.28,
  });
  const trimMat = new THREE.MeshStandardMaterial({ color: trimColor, roughness: 0.85, side: THREE.DoubleSide });
  const a = new THREE.Mesh(leather, leatherMat);
  const b = new THREE.Mesh(trim, trimMat);
  a.castShadow = b.castShadow = true;
  g.add(a, b);
  g.userData.leatherMat = leatherMat;
  return g;
}
