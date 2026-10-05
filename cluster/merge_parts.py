#!/usr/bin/env python
"""Merge the part_*_hits.root files of one cluster production into a single file
that plot_arrival_times.py / classifier.py can read, renumbering eventIDs so they
stay unique. Uses uproot only (no ROOT needed).

usage: python merge_parts.py out/cluster/em_E50_d20 [--out out/em_E50_d20_hits.root]
"""
import argparse, glob, json, os
import numpy as np
import uproot

ap = argparse.ArgumentParser()
ap.add_argument("part_dir")
ap.add_argument("--out", default=None)
a = ap.parse_args()
parts = sorted(glob.glob(os.path.join(a.part_dir, "part_*_hits.root")))
if not parts:
    raise SystemExit(f"no part_*_hits.root in {a.part_dir}")
name = os.path.basename(os.path.normpath(a.part_dir))
out = a.out or os.path.join(os.path.dirname(os.path.normpath(a.part_dir)), os.pardir, f"{name}_hits.root")

hits, events, offset, n_ev_total = [], [], 0, 0
for p in parts:
    f = uproot.open(p)
    h = f["PhotonHits"].arrays(library="np")
    ev = f["Events"].arrays(library="np") if "Events" in f else None
    args = json.load(open(p.replace("_hits.root", "_args.json")))
    n_ev = int(args["numevents"])
    h["eventID"] = h["eventID"] + offset
    hits.append(h)
    if ev is not None:
        ev["eventID"] = ev["eventID"] + offset
        events.append(ev)
    offset += n_ev; n_ev_total += n_ev

def cat(arrays):
    """Concatenate; ROOT string branches come back as object arrays, which uproot
    cannot write, so hand those over as awkward string arrays."""
    import awkward as ak
    a = np.concatenate(arrays)
    return ak.Array([str(x) for x in a]) if a.dtype == object else a

merged = {k: cat([h[k] for h in hits]) for k in hits[0]}
with uproot.recreate(out) as f:
    f["PhotonHits"] = merged
    if events:
        f["Events"] = {k: cat([e[k] for e in events]) for k in events[0]}
json.dump(dict(args, numevents=n_ev_total, merged_parts=len(parts)), open(out.replace("_hits.root", "_args.json"), "w"), indent=1)
print(f"{len(parts)} parts, {n_ev_total} events, {len(merged['eventID'])} hits -> {out}")
