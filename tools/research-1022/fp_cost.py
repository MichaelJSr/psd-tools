"""fp_cost.py <psd>...: cost of a content fingerprint of every layer (record bytes + channel bytes)."""
import io, sys, time, hashlib
from psd_tools import PSDImage

def fp_all(psd):
    memo = {}
    def fp(layer):
        h = hashlib.blake2b(digest_size=16)
        b = io.BytesIO(); layer._record.write(b); h.update(b.getvalue())
        for ch in layer._channels: h.update(ch.data)
        if layer.is_group():
            br = getattr(layer, "_bounding_record", None)
            if br is not None:
                b = io.BytesIO(); br.write(b); h.update(b.getvalue())
            for c in layer: h.update(fp(c))
        d = h.digest(); memo[id(layer)] = d; return d
    for l in psd: fp(l)
    return memo

for p in sys.argv[1:]:
    t = time.perf_counter(); psd = PSDImage.open(p); t_open = time.perf_counter() - t
    t = time.perf_counter(); m = fp_all(psd); t_fp = time.perf_counter() - t
    t = time.perf_counter(); m = fp_all(psd); t_fp2 = time.perf_counter() - t
    nbytes = sum(len(ch.data) for l in psd.descendants() for ch in l._channels)
    print(p.rsplit('/',1)[1], 'layers', len(m), 'open %.3f fp %.3f fp2 %.3f' % (t_open, t_fp, t_fp2), 'chan MB %.1f' % (nbytes/2**20))
