"""Build 0-data/AlLi-candidates-1000.pkl.gz -- the candidate pool for data selection.

Prepared OFFLINE (the tutorial ships the result). The fps-selection variant of the tutorial
picks 128 of these with `grace_uq select --strategy fps-all` in the foundation model's feature
space and adds them to the 112 hull training structures.

The pool is a uniformly random draw from the source database after the same hygiene filters as
the extended evaluation set (make_extended_eval.py), with everything the tutorials already use
removed: the hull train/test sets, the whole OOD prototype family and the 265 evaluation
structures. Uniform means roughly half of the pool is pure Al or pure Li, exactly as in the
database; the selection plots in the notebook show what fps does with such a lopsided pool.
"""
import numpy as np, pandas as pd

SRC = "/home/users/lysogy36/acefit/Al-Li/8a-LADDER-k0.3/AlLi_BIG_Oct2023_UNIQUE_free-e.pkl.gz"
OUT = "0-data/AlLi-candidates-1000.pkl.gz"
N_CANDIDATES, NAT_MAX = 1000, 24
MAX_HULL_DIST = 0.500      # eV/atom: stay in the physically relevant window
SEED = 0

B = pd.read_pickle(SRC, compression="gzip")
used = set()
for f in ("AlLi-hull-train", "AlLi-hull-test", "AlLi-ood-mp1191737", "AlLi-extended-eval"):
    used |= set(pd.read_pickle(f"0-data/{f}.pkl.gz", compression="gzip")["name"].astype(str))

n = B["name"].astype(str)
B["nat"]  = B["ase_atoms"].map(len)
B["epa"]  = B["energy"]/B["nat"]
B["fmax"] = B["forces"].map(lambda f: np.abs(np.asarray(f)).max())
B["dmin"] = [float(a.get_all_distances(mic=True)[np.triu_indices(len(a), 1)].min())
             if len(a) > 1 else 99.0 for a in B["ase_atoms"]]
B["c_Li"] = [sum(s == "Li" for s in a.get_chemical_symbols())/len(a) for a in B["ase_atoms"]]
B["dft"]  = n.str.contains("/DFT/")
B["kind"] = np.where(B["dft"], n.str.extract(r"/DFT/[^/]+/([^/]+)/")[0].str.replace(r"\d+$", "", regex=True), "generated")

ok = ((~n.isin(used))                    # not in training, test, the OOD demo or the evaluation set
      & ~n.str.contains("mp-1191737")    # keep the OOD family out entirely
      & ~(B.dft & (B.kind == "nn"))      # nn scans: isolated atoms beyond the 6 A cutoff
      & (B.dmin >= 2.0) & (B.dmin <= 6.0)   # no hard overlaps, no isolated atoms
      & (B.epa <= 0) & (B.fmax <= 10.0)     # no multi-eV / exploding-force outliers
      & (B.nat >= 2) & (B.nat <= NAT_MAX))  # small cells: cheap to predict and to train on

# hull distance on the tutorial's own DFT reference, so the window is physically meaningful
from amstools.thermodynamics import (ensure_energy_per_atom_column, compute_compositions,
                                     compute_formation_energy, compute_convexhull_dist)
ref  = pd.read_pickle("0-data/AlLi-hull-all.pkl.gz", compression="gzip")
elig = B[ok]                                                 # eligible rows, original index kept
both = pd.concat([ref, elig], ignore_index=True)             # reference rows first, then the eligible rows in order
ensure_energy_per_atom_column(both, energy_column="energy")
compute_compositions(both)
compute_formation_energy(both, verbose=False)
compute_convexhull_dist(both, verbose=False)
# select by ROW, not by name: names repeat in the database, so a name-based filter would let the
# high-energy twin of an accepted structure through
dist = both["e_chull_dist_per_atom"].values[len(ref):]      # aligned with elig, row by row
B.loc[elig.index, "hull_dist"] = dist
ok   = B.index.isin(elig.index[dist <= MAX_HULL_DIST])
print(f"eligible after all filters: {ok.sum()} of {len(B)}")

cand = (B[ok].sample(n=N_CANDIDATES, random_state=SEED)
         [["name", "ase_atoms", "energy", "forces", "nat", "c_Li", "fmax", "hull_dist", "kind"]]
         .reset_index(drop=True))
assert not set(cand["name"].astype(str)) & used, "candidate overlaps a set already in use"
assert cand["hull_dist"].max() <= MAX_HULL_DIST, "hull-distance filter failed"
# names in the source database repeat (generated cells reuse `structure_N` across generations);
# the tools key on `name`, so give every candidate a unique id and keep the original alongside
cand["source_name"] = cand["name"].astype(str)
cand["name"] = [f"cand_{i:04d}" for i in range(len(cand))]
print(f"  source names repeated within the pool: {cand.source_name.duplicated().sum()}")
cand.to_pickle(OUT, compression="gzip")

print(f"{OUT}: {len(cand)} structures, {cand.nat.sum()} atoms")
print(f"  nat {cand.nat.min()}..{cand.nat.max()} (median {cand.nat.median():.0f})")
print(f"  |F|max median {cand.fmax.median():.3f}, p90 {cand.fmax.quantile(.9):.2f}, max {cand.fmax.max():.2f} eV/A")
print(f"  pure Al {(cand.c_Li == 0).sum()}, pure Li {(cand.c_Li == 1).sum()}, alloys {((cand.c_Li > 0) & (cand.c_Li < 1)).sum()}")
print("  by kind:", cand.kind.value_counts().to_dict())
