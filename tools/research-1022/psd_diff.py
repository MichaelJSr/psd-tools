"""psd_diff.py <old.psd> <new.psd>: what changed between two versions of a card, layer by layer.

Layers pair by path (group names / layer name, nth duplicate); each pair compares
record fields and content hashes (rcache.layer_key, so pixels, text, effects and
masks all count) and says what kind of change it is."""
import difflib, hashlib, io, sys
from psd_tools import PSDImage
from psd_tools.constants import Tag

def path_map(psd):
    out, seen = {}, {}
    def walk(g, prefix):
        for l in g:
            p = prefix + (l.name,)
            n = seen.get(p, 0); seen[p] = n + 1
            out[p + ((n,) if n else ())] = l
            if l.is_group():
                walk(l, p)
    walk(psd, ())
    return out

def blob(x):
    b = io.BytesIO(); x.write(b); return b.getvalue()

def h(data):
    return hashlib.blake2b(data, digest_size=8).hexdigest()

def block(l, tag):
    b = l._record.tagged_blocks.get(tag) if l._record.tagged_blocks is not None else None
    return None if b is None else h(blob(b))

def diff(a, b):
    notes = []
    if a.visible != b.visible: notes.append("shown" if b.visible else "hidden")
    if a.kind != b.kind: notes.append(f"kind {a.kind} -> {b.kind}")
    if not a.is_group() and a.bbox != b.bbox:
        if (a.width, a.height) == (b.width, b.height):
            notes.append(f"moved {b.left - a.left:+d},{b.top - a.top:+d} px")
        else:
            notes.append(f"box {a.bbox} -> {b.bbox}")
    if a.opacity != b.opacity: notes.append(f"opacity {a.opacity} -> {b.opacity}")
    if a.blend_mode != b.blend_mode: notes.append(f"blend {a.blend_mode.name} -> {b.blend_mode.name}")
    if a.kind == "type" and b.kind == "type":
        ta, tb = a.text.replace("\r", "\n"), b.text.replace("\r", "\n")
        if ta != tb:
            sm = difflib.SequenceMatcher(None, ta, tb)
            ch = [f"{ta[i1:i2]!r}->{tb[j1:j2]!r}" for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
            notes.append("text " + "; ".join(ch[:6]) + (" ..." if len(ch) > 6 else ""))
        elif block(a, Tag.TYPE_TOOL_OBJECT_SETTING) != block(b, Tag.TYPE_TOOL_OBJECT_SETTING):
            notes.append("text style (fonts, sizes, colours or runs)")
    for tag, what in ((Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, "layer style"), (Tag.VECTOR_MASK_SETTING1, "vector mask"),
                      (Tag.VECTOR_MASK_SETTING2, "vector mask")):
        if block(a, tag) != block(b, tag): notes.append(what + " changed")
    if not a.is_group():
        ca = h(b"".join(c.data for c in a._channels)); cb = h(b"".join(c.data for c in b._channels))
        if ca != cb and a.kind != "type": notes.append("pixels changed")
    return notes

old, new = PSDImage.open(sys.argv[1]), PSDImage.open(sys.argv[2])
A, B = path_map(old), path_map(new)
for p in A:
    if p not in B: print("removed  ", "/".join(map(str, p)))
for p in B:
    if p not in A: print("added    ", "/".join(map(str, p)), f"({B[p].kind}, {'shown' if B[p].visible else 'hidden'})")
for p in A:
    if p in B:
        n = diff(A[p], B[p])
        if n: print("changed  ", "/".join(map(str, p)) + ":", ", ".join(n))
