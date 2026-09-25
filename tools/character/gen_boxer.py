"""
Generates a realistic, game-ready boxer with MakeHuman (MPFB) inside Blender.

Run headless with the SSD-local toolchain:
  /Volumes/WorkPlace/Tools/BlenderBoxing/blender.sh --background --factory-startup? (no: needs MPFB prefs)
  /Volumes/WorkPlace/Tools/BlenderBoxing/blender.sh --background --python tools/character/gen_boxer.py -- <preset> <out.glb>

Presets are defined in PRESETS below. Everything is CC0 (MakeHuman assets).

Pipeline
  1. body: MakeHuman macro sliders (young athletic male) + V-taper/pec/lat targets
  2. low-poly muscular proxy body (game-ready topology)
  3. realistic skin, eyes, brows, lashes, teeth, short hair (GAMEENGINE materials)
  4. Mixamo-compatible skeleton (the game's retargeting expects mixamorig:* bones)
  5. boxing trunks (+ waistband) and high-top boots grown from the body surface,
     so they inherit its skin weights and bend correctly; skin hidden under the
     kit is removed to prevent poke-through
  6. textures downsized, GLB export (no animations, no morphs)
"""
import os
import sys

import bmesh
import bpy
from mathutils import Vector

MPFB = "bl_ext.user_default.mpfb"
HumanService = __import__(f"{MPFB}.services.humanservice", fromlist=["HumanService"]).HumanService
TargetService = __import__(f"{MPFB}.services.targetservice", fromlist=["TargetService"]).TargetService
LocationService = __import__(f"{MPFB}.services.locationservice", fromlist=["LocationService"]).LocationService

DATA = LocationService.get_user_data()
MPFB_DIR = os.path.dirname(os.path.dirname(sys.modules[f"{MPFB}.services.targetservice"].__file__))
ClothesService = __import__(f"{MPFB}.services.clothesservice", fromlist=["ClothesService"]).ClothesService

# Facial expressions, composed from MakeHuman's FACS-like "expression units".
# Each becomes a glTF morph target the game blends at runtime.
EXPRESSIONS = {
    "focus": {"eyebrows-left-down": 0.55, "eyebrows-right-down": 0.55, "eye-left-slit": 0.25, "eye-right-slit": 0.25,
              "mouth-compression": 0.3},
    "effort": {"eyebrows-left-down": 0.7, "eyebrows-right-down": 0.7, "eye-left-slit": 0.5, "eye-right-slit": 0.5,
               "mouth-compression": 0.5, "mouth-retraction": 0.35, "nose-left-dilatation": 0.5,
               "nose-right-dilatation": 0.5},
    "pain": {"eye-left-slit": 0.9, "eye-right-slit": 0.9, "eyebrows-left-down": 0.6, "eyebrows-right-down": 0.6,
             "eyebrows-left-inner-up": 0.5, "eyebrows-right-inner-up": 0.5, "nose-left-elevation": 0.6,
             "nose-right-elevation": 0.6, "mouth-retraction": 0.6, "mouth-upward-retraction": 0.4, "mouth-open": 0.25},
    "exhale": {"mouth-pursing": 0.75, "mouth-compression": 0.2, "nose-left-dilatation": 0.4,
               "nose-right-dilatation": 0.4},
    "mouthOpen": {"mouth-open": 0.9},
    "blink": {"eye-left-closure": 1.0, "eye-right-closure": 1.0},
    "ko": {"eye-left-closure": 0.92, "eye-right-closure": 0.92, "mouth-open": 0.5, "eyebrows-left-inner-up": 0.35,
           "eyebrows-right-inner-up": 0.35},
    "smile": {"mouth-corner-puller": 0.85, "eye-left-slit": 0.2, "eye-right-slit": 0.2, "mouth-open": 0.12},
}

PRESETS = {
    # the player: light-skinned, blue trunks
    "player": {
        "macro": {"gender": 1.0, "age": 0.45, "muscle": 0.88, "weight": 0.42, "proportions": 0.95, "height": 0.6,
                  "race": {"caucasian": 0.8, "asian": 0.1, "african": 0.1}},
        "targets": {"torso-vshape-incr": 0.7, "torso-muscle-pectoral-incr": 0.5, "torso-muscle-dorsi-incr": 0.6,
                    "measure-neck-circ-incr": 0.6},
        "skin": "skins/young_caucasian_male/young_caucasian_male.mhmat",
        "face_units": "caucasian",
        "eyes": "brown",
        "hair": "hair/short02/short02.mhclo",
        "eyebrows": "eyebrows/eyebrow001/eyebrow001.mhclo",
        "trunks": (0.07, 0.2, 0.75), "band": (0.95, 0.95, 0.95),
        "boots": (0.92, 0.92, 0.93), "sole": (0.08, 0.08, 0.09),
    },
    # the opponent: dark-skinned, heavier build, red trunks
    "opponent": {
        "macro": {"gender": 1.0, "age": 0.55, "muscle": 0.95, "weight": 0.52, "proportions": 0.9, "height": 0.62,
                  "race": {"caucasian": 0.05, "asian": 0.05, "african": 0.9}},
        "targets": {"torso-vshape-incr": 0.8, "torso-muscle-pectoral-incr": 0.7, "torso-muscle-dorsi-incr": 0.7,
                    "measure-neck-circ-incr": 0.9},
        "skin": "skins/young_african_male/young_african_male.mhmat",
        "face_units": "african",
        "eyes": "brownlight",
        "hair": "hair/short04/short04.mhclo",
        "eyebrows": "eyebrows/eyebrow006/eyebrow006.mhclo",
        "trunks": (0.62, 0.04, 0.06), "band": (0.95, 0.78, 0.25),
        "boots": (0.05, 0.05, 0.06), "sole": (0.7, 0.7, 0.72),
    },
}


def log(*a):
    print("[gen_boxer]", *a, flush=True)


def data(rel):
    p = os.path.join(DATA, rel)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.armatures):
        for b in list(block):
            if b.users == 0:
                block.remove(b)


def principled(name, color, rough=0.5, metal=0.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if coat and "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
        bsdf.inputs["Coat Roughness"].default_value = 0.3
    return m


def dominant_group(obj, v, names_by_index):
    best, bw = None, 0.0
    for g in v.groups:
        n = names_by_index.get(g.group)
        if n and g.weight > bw:
            best, bw = n, g.weight
    return best


def boundary_verts(bm):
    return {v for e in bm.edges if e.is_boundary for v in e.verts}


def grow_garment(body, name, keep_vert, offset_fn, mat_fn, materials, hems=None, band=None):
    """Duplicate `body`, keep region verts, push them out along normals.

    hems: list of (predicate(co), z) — open-edge verts matching the predicate
          are snapped to height z, giving clean straight hems instead of the
          saw-tooth left by cutting along the mesh grid.
    band: (height, extra_offset, material_index) — builds a separate elastic
          waistband strip hanging from the top opening.
    """
    garment = body.copy()
    garment.data = body.data.copy()
    garment.name = name
    garment.data.name = name
    bpy.context.collection.objects.link(garment)
    garment.shape_key_clear()  # kit doesn't pull faces
    garment.data.materials.clear()
    for m in materials:
        garment.data.materials.append(m)

    names = {g.index: g.name for g in garment.vertex_groups}
    bm = bmesh.new()
    bm.from_mesh(garment.data)
    bm.verts.ensure_lookup_table()
    deform = bm.verts.layers.deform.active
    drop = []
    for v in bm.verts:
        groups = {names[i]: w for i, w in v[deform].items()} if deform else {}
        dom = max(groups, key=groups.get) if groups else None
        if not keep_vert(v.co, dom):
            drop.append(v)
    bmesh.ops.delete(bm, geom=drop, context="VERTS")
    # clean hems: snap the open edges to straight horizontal lines
    for pred, z in hems or []:
        for v in boundary_verts(bm):
            if pred(v.co):
                v.co.z = z
    bm.normal_update()
    for v in bm.verts:
        v.co += v.normal * offset_fn(v.co)
    for f in bm.faces:
        f.material_index = mat_fn(f.calc_center_median())
        f.smooth = True
    if band:
        height, extra, mat_index = band
        top_z = max(v.co.z for v in bm.verts)
        top = [e for e in bm.edges if e.is_boundary and all(v.co.z > top_z - 0.01 for v in e.verts)]
        made = {}

        def below(v):
            if v not in made:
                out = Vector((v.normal.x, v.normal.y, 0.0))
                out = out.normalized() if out.length > 1e-6 else out
                nv = bm.verts.new(v.co + out * extra - Vector((0, 0, height)))
                if deform:
                    for k, w in v[deform].items():
                        nv[deform][k] = w
                v.co += out * extra
                made[v] = nv
            return made[v]

        for e in top:
            a, b = e.verts
            f = bm.faces.new((a, b, below(b), below(a)))
            f.material_index = mat_index
            f.smooth = True
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(garment.data)
    bm.free()
    return garment


def remove_hidden_skin(body, hidden_fn):
    names = {g.index: g.name for g in body.vertex_groups}
    bm = bmesh.new()
    bm.from_mesh(body.data)
    deform = bm.verts.layers.deform.active
    faces = []
    for f in bm.faces:
        ok = True
        for v in f.verts:
            groups = {names[i]: w for i, w in v[deform].items()} if deform else {}
            dom = max(groups, key=groups.get) if groups else None
            if not hidden_fn(v.co, dom):
                ok = False
                break
        if ok:
            faces.append(f)
    bmesh.ops.delete(bm, geom=faces, context="FACES_ONLY")
    bm.to_mesh(body.data)
    bm.free()
    log("removed hidden skin faces:", len(faces))


def apply_modifiers(obj, keep=("ARMATURE",)):
    """Bake non-armature modifiers (masks etc.) so morph targets line up."""
    for m in list(obj.modifiers):
        if m.type in keep:
            continue
        with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
            try:
                bpy.ops.object.modifier_apply(modifier=m.name)
            except Exception as e:
                log("could not apply", obj.name, m.name, e)
                obj.modifiers.remove(m)


def bake_expressions(basemesh, assets, race):
    """For each expression: pose the MakeHuman base head with expression
    units, refit the game assets (body proxy, brows, lashes, teeth) to it and
    store the result as a shape key → glTF morph target.

    All fitting happens BEFORE any shape key exists on the assets: once an
    asset has shape keys, MPFB's refit writes into its Basis key instead."""
    units_dir = os.path.join(MPFB_DIR, "data", "targets", "expression", "units", race)
    for obj in assets:
        obj.shape_key_clear()
    neutral = {obj.name: [v.co.copy() for v in obj.data.vertices] for obj in assets}
    posed = {}  # expr -> {obj name -> [co]}
    for expr, units in EXPRESSIONS.items():
        names = []
        for unit, w in units.items():
            path = os.path.join(units_dir, unit + ".target.gz")
            if not os.path.exists(path):
                log("missing expression unit", unit)
                continue
            name = f"expr.{expr}.{unit}"
            TargetService.load_target(basemesh, path, weight=w, name=name)
            names.append(name)
        posed[expr] = {}
        for obj in assets:
            ClothesService.fit_clothes_to_human(obj, basemesh, set_parent=False)
            posed[expr][obj.name] = [v.co.copy() for v in obj.data.vertices]
            for i, v in enumerate(obj.data.vertices):
                v.co = neutral[obj.name][i]
        for n in names:
            kb = basemesh.data.shape_keys.key_blocks.get(n)
            if kb:
                basemesh.shape_key_remove(kb)
        moved = max((a - b).length for o in assets for a, b in zip(posed[expr][o.name], neutral[o.name]))
        log(f"expression {expr}: {len(names)} units, max move {moved * 100:.2f} cm")
    # now create the shape keys from the recorded positions
    for obj in assets:
        obj.shape_key_add(name="Basis", from_mix=False)
        for expr in EXPRESSIONS:
            key = obj.shape_key_add(name=expr, from_mix=False)
            for i, co in enumerate(posed[expr][obj.name]):
                key.data[i].co = co
        obj.data.update()


def downsize_images(max_px):
    for img in bpy.data.images:
        if img.size[0] > max_px or img.size[1] > max_px:
            s = max_px / max(img.size)
            img.scale(int(img.size[0] * s), int(img.size[1] * s))
            log("resized", img.name, tuple(img.size))


def build(preset_name, out_path):
    P = PRESETS[preset_name]
    clear_scene()

    # 1. body shape
    basemesh = HumanService.create_human(macro_detail_dict=P["macro"], feet_on_ground=True, scale=0.1)
    for t, w in P["targets"].items():
        TargetService.load_target(basemesh, TargetService.target_full_path(t), weight=w)
    log("basemesh", basemesh.name)

    # skeleton first, so every asset added afterwards is weighted to it
    HumanService.add_builtin_rig(basemesh, "mixamo", import_weights=True)
    next(o for o in bpy.data.objects if o.type == "ARMATURE").name = "Armature"

    # 2. game-ready muscular proxy
    proxy = HumanService.add_mhclo_asset(data("proxymeshes/male_muscle_13290/male_muscle_13290.proxy"), basemesh,
                                         asset_type="Proxymeshes", subdiv_levels=0, material_type="GAMEENGINE")

    # 3. skin / face assets
    HumanService.set_character_skin(data(P["skin"]), basemesh, bodyproxy=proxy, skin_type="GAMEENGINE",
                                    material_instances=False)
    HumanService.add_mhclo_asset(data("eyes/low-poly/low-poly.mhclo"), basemesh, asset_type="Eyes",
                                        subdiv_levels=0, material_type="GAMEENGINE")
    for rel, kind in ((P["eyebrows"], "Eyebrows"), ("eyelashes/eyelashes01/eyelashes01.mhclo", "Eyelashes"),
                      ("teeth/teeth_base/teeth_base.mhclo", "Teeth"), (P["hair"], "Hair")):
        HumanService.add_mhclo_asset(data(rel), basemesh, asset_type=kind, subdiv_levels=0, material_type="GAMEENGINE")

    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    bones = rig.data.bones

    # facial expressions → morph targets on everything that sits on the face
    face_assets = [o for o in bpy.data.objects if o.type == "MESH" and o is not basemesh
                   and any(k in o.name.lower() for k in ("muscle", "eyebrow", "eyelash", "teeth"))]
    if not os.environ.get("NO_EXPR"):
        for o in face_assets:
            apply_modifiers(o)
        bake_expressions(basemesh, face_assets, P["face_units"])

    def bz(name):
        return (rig.matrix_world @ bones[f"mixamorig:{name}"].head_local).z

    hips, spine, thigh, knee, foot = bz("Hips"), bz("Spine"), bz("LeftUpLeg"), bz("LeftLeg"), bz("LeftFoot")
    trunk_top = spine + 0.035
    trunk_bot = thigh - 0.42 * (thigh - knee)
    crotch = thigh - 0.07
    boot_top = foot + 0.5 * (knee - foot)
    log("landmarks", dict(hips=hips, spine=spine, thigh=thigh, knee=knee, foot=foot))

    trunk_groups = {"mixamorig:Hips", "mixamorig:Spine", "mixamorig:LeftUpLeg", "mixamorig:RightUpLeg"}
    boot_groups = {"mixamorig:LeftLeg", "mixamorig:RightLeg", "mixamorig:LeftFoot", "mixamorig:RightFoot",
                   "mixamorig:LeftToeBase", "mixamorig:RightToeBase"}

    # 5. kit
    satin = principled("Trunks", P["trunks"], rough=0.28, metal=0.05, coat=0.35)
    band = principled("TrunksBand", P["band"], rough=0.4)
    leather = principled("Boots", P["boots"], rough=0.42, coat=0.2)
    sole = principled("BootSole", P["sole"], rough=0.8)

    def trunks_keep(co, dom):
        return trunk_bot <= co.z <= trunk_top and dom in trunk_groups

    def trunks_offset(co):
        base = 0.007
        if co.z < crotch:  # loose, flared legs like real boxing trunks
            base += 0.028 * (crotch - co.z) / max(1e-3, crotch - trunk_bot)
        if co.z > trunk_top - 0.06:  # elastic waistband sits proud
            base += 0.004
        return base

    grow_garment(proxy, "Trunks", trunks_keep, trunks_offset, lambda c: 0, [satin, band],
                 hems=[(lambda co: co.z > trunk_top - 0.09, trunk_top), (lambda co: co.z < crotch, trunk_bot)],
                 band=(0.06, 0.006, 1))

    def boots_keep(co, dom):
        return co.z <= boot_top and dom in boot_groups

    grow_garment(proxy, "Boots", boots_keep, lambda co: 0.006 + (0.004 if co.z < 0.03 else 0.0),
                 lambda c: 1 if c.z < 0.025 else 0, [leather, sole],
                 hems=[(lambda co: co.z > boot_top - 0.08, boot_top)])

    # skin under the kit is never visible: remove it (no poke-through, fewer tris)
    remove_hidden_skin(proxy, lambda co, dom: (trunk_bot + 0.03 < co.z < trunk_top - 0.03 and dom in trunk_groups)
                       or (co.z < boot_top - 0.03 and dom in boot_groups))

    # 6. export
    basemesh.hide_set(True)
    basemesh.hide_render = True
    downsize_images(2048)
    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.data.objects:
        if o is basemesh:
            continue
        if o.type in {"MESH", "ARMATURE"}:
            o.select_set(True)
            # garments / assets may carry subdivision modifiers: drop them
            for m in list(o.modifiers):
                if m.type == "SUBSURF":
                    o.modifiers.remove(m)
    bpy.context.view_layer.objects.active = rig

    def export(path, morphs):
        bpy.ops.export_scene.gltf(
            filepath=path,
            export_format="GLB",
            use_selection=True,
            export_apply=True,
            export_skins=True,
            export_animations=False,
            export_morph=morphs,
            export_morph_normal=False,
            export_image_format="JPEG",
            export_jpeg_quality=85,
            export_yup=True,
        )

    # Two exports: the morph targets come from one, the base mesh from the
    # other. Blender's exporter computes normals differently for meshes with
    # shape keys (flipping a few around the nostrils), so the clean base is
    # exported without shape keys and optimize.mjs grafts the morphs onto it.
    export(out_path.replace(".glb", ".morphs.glb"), True)
    for o in bpy.data.objects:
        if o.type == "MESH" and o.data.shape_keys and o is not basemesh:
            o.shape_key_clear()
    export(out_path, False)
    tris = sum(len(o.data.loop_triangles) for o in bpy.data.objects
               if o.type == "MESH" and o.select_get() and (o.data.calc_loop_triangles() or True))
    log("exported", out_path, "triangles≈", tris, "bytes", os.path.getsize(out_path))


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    preset = argv[0] if argv else "player"
    out = os.path.abspath(argv[1] if len(argv) > 1 else f"{preset}.glb")
    build(preset, out)
