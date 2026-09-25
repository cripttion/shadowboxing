// Post-process a generated boxer GLB for real-time use on low-end devices:
//  * correct alpha modes (skin/eyes/teeth opaque; hair/brows/lashes cut-out)
//  * per-material texture budgets, re-encoded as WebP
//  * dedupe / prune / weld
// usage: node tools/character/optimize.mjs in.glb out.glb
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { dedup, prune, textureCompress } from '@gltf-transform/functions';
import sharp from 'sharp';
import { existsSync } from 'node:fs';

const [input, output] = process.argv.slice(2);
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const doc = await io.read(input);
const root = doc.getRoot();

// Graft facial-expression morph targets from the companion export
// (<name>.morphs.glb): same meshes, same vertex order, but its base normals
// are unreliable — so only the targets are taken from it.
const morphFile = input.replace(/\.glb$/, '.morphs.glb');
if (existsSync(morphFile)) {
  const src = await io.read(morphFile);
  const srcMeshes = new Map(src.getRoot().listMeshes().map((m) => [m.getName(), m]));
  let grafted = 0;
  for (const mesh of root.listMeshes()) {
    const from = srcMeshes.get(mesh.getName());
    if (!from) continue;
    const dstPrims = mesh.listPrimitives();
    const srcPrims = from.listPrimitives();
    if (!srcPrims[0] || !srcPrims[0].listTargets().length) continue;
    dstPrims.forEach((prim, pi) => {
      const sp = srcPrims[pi];
      if (!sp || sp.getAttribute('POSITION').getCount() !== prim.getAttribute('POSITION').getCount()) return;
      const names = from.getExtras().targetNames || [];
      for (const [ti, t] of sp.listTargets().entries()) {
        const acc = t.getAttribute('POSITION');
        // sparse: expressions only move face vertices, so store just those
        const copy = doc.createAccessor().setType('VEC3').setArray(acc.getArray().slice()).setBuffer(root.listBuffers()[0]).setSparse(true);
        // glTF-Transform writes extras.targetNames from the targets' own names
        prim.addTarget(doc.createPrimitiveTarget(names[ti] || `morph${ti}`).setAttribute('POSITION', copy));
      }
    });
    mesh.setWeights(new Array(srcPrims[0].listTargets().length).fill(0));
    grafted++;
  }
  console.log(`  grafted expression morphs onto ${grafted} meshes`);
}

// max texture size per material (by name)
const BUDGET = [
  [/body|skin/i, 2048],
  [/short|hair/i, 1024],
  [/eye(?!brow|lash)|low-poly/i, 512],
  [/teeth/i, 256],
  [/brow|lash/i, 256],
];

for (const m of root.listMaterials()) {
  const n = m.getName();
  if (/brow|lash/i.test(n)) {
    m.setAlphaMode('BLEND');
  } else if (/short|hair/i.test(n)) {
    // hair cards: alpha-tested so they sort correctly without blending
    m.setAlphaMode('MASK');
    m.setAlphaCutoff(0.45);
  } else {
    m.setAlphaMode('OPAQUE');
  }
  if (/body/i.test(n)) m.setName('Skin');
  m.setDoubleSided(/short|hair|brow|lash/i.test(n));
}

// no weld(): merging vertices that coincide at rest but differ in morph targets
// corrupts the face (seen as a stray flap at the nostril)
await doc.transform(dedup(), prune());

for (const t of root.listTextures()) {
  const users = t.listParents().filter((p) => p.propertyType === 'Material').map((m) => m.getName());
  const name = users.join(',');
  const max = (BUDGET.find(([re]) => re.test(name)) || [null, 1024])[1];
  const [w, h] = t.getSize();
  if (w > max || h > max) {
    const img = await sharp(Buffer.from(t.getImage())).resize(max, max, { fit: 'inside' }).png().toBuffer();
    t.setImage(new Uint8Array(img));
  }
}
await doc.transform(textureCompress({ encoder: sharp, targetFormat: 'webp', quality: 86 }));

await io.write(output, doc);
for (const t of root.listTextures()) {
  const users = t.listParents().filter((p) => p.propertyType === 'Material').map((m) => m.getName());
  console.log('  tex', users.join(','), t.getMimeType(), t.getSize().join('x'), `${(t.getImage().byteLength / 1024) | 0}KB`);
}
const { size } = await import('node:fs').then((fs) => fs.promises.stat(output));
console.log(`${output}: ${(size / 1024 / 1024).toFixed(2)} MB`);
