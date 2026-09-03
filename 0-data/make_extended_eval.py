"""Build 0-data/AlLi-extended-eval.pkl.gz -- a held-out DFT evaluation set.

Prepared OFFLINE (the tutorial ships the result). Broader than the 128-structure
train+test set: many more deformation types and many more compositions, with the
train/test/OOD structures and the known pathologies removed.
"""
import numpy as np, pandas as pd

SRC = "/home/users/lysogy36/acefit/Al-Li/8a-LADDER-k0.3/AlLi_BIG_Oct2023_UNIQUE_free-e.pkl.gz"
OUT = "0-data/AlLi-extended-eval.pkl.gz"
N_DFT, N_GEN, NAT_MAX = 200, 200, 24
MAX_HULL_DIST = 0.500      # eV/atom: stay in the physically relevant window
MAX_PER_COMP  = 30         # keep pure Al / pure Li from dominating
RNG = np.random.default_rng(0)

B = pd.read_pickle(SRC, compression="gzip")
used = set()
for f in ("AlLi-hull-train", "AlLi-hull-test", "AlLi-ood-mp1191737"):
    used |= set(pd.read_pickle(f"0-data/{f}.pkl.gz", compression="gzip")["name"].astype(str))

n = B["name"].astype(str)
B["nat"]  = B["ase_atoms"].map(len)
B["epa"]  = B["energy"]/B["nat"]
B["fmax"] = B["forces"].map(lambda f: np.abs(np.asarray(f)).max())
B["dmin"] = [float(a.get_all_distances(mic=True)[np.triu_indices(len(a), 1)].min())
             if len(a) > 1 else 99.0 for a in B["ase_atoms"]]
B["c_Li"] = [sum(s == "Li" for s in a.get_chemical_symbols())/len(a) for a in B["ase_atoms"]]
B["dft"]  = n.str.contains("/DFT/")
B["kind"] = n.str.extract(r"/DFT/[^/]+/([^/]+)/")[0]

ok = ((~n.isin(used))                    # never seen in training, test or the OOD demo
      & ~n.str.contains("mp-1191737")    # keep the OOD family out of the general metric
      & ~(B.dft & (B.kind == "nn"))      # nn scans: isolated atoms beyond the 6 A cutoff
      & (B.dmin >= 2.0) & (B.dmin <= 6.0)   # no hard overlaps, no isolated atoms
      & (B.epa <= 0) & (B.fmax <= 10.0)     # no multi-eV / exploding-force outliers
      & (B.nat >= 2) & (B.nat <= NAT_MAX))  # small cells: the set stays cheap to evaluate

# hull distance on the tutorial's own DFT reference, so the window is physically meaningful
from amstools.thermodynamics import (ensure_energy_per_atom_column, compute_compositions,
                                     compute_formation_energy, compute_convexhull_dist)
ref = pd.read_pickle("0-data/AlLi-hull-all.pkl.gz", compression="gzip")
both = pd.concat([ref.assign(_ext=False), B[ok].assign(_ext=True)], ignore_index=True)
for fn in (lambda d: ensure_energy_per_atom_column(d, energy_column="energy"), compute_compositions,
           lambda d: compute_formation_energy(d, verbose=False),
           lambda d: compute_convexhull_dist(d, verbose=False)):
    fn(both)
keep = set(both.loc[both._ext & (both.e_chull_dist_per_atom <= MAX_HULL_DIST), "name"].astype(str))
ok &= n.isin(keep)

def take(df, by, n_total):
    """Round-robin across groups so no single deformation type or composition dominates."""
    groups = [g.sample(frac=1, random_state=0) for _, g in df.groupby(by)]
    out, i = [], 0
    while sum(len(x) for x in out) < n_total and any(len(g) > i for g in groups):
        for g in groups:
            if i < len(g): out.append(g.iloc[[i]])
            if sum(len(x) for x in out) >= n_total: break
        i += 1
    return pd.concat(out).head(n_total)

d = take(B[ok & B.dft],  "kind", N_DFT)                       # deformation types
g = B[ok & ~B.dft].copy()
g["cbin"] = (g["c_Li"]*20).round().astype(int)                # 5 mol-% composition bins
g = take(g, "cbin", N_GEN)

ext = pd.concat([d, g], ignore_index=True)
ext = (ext.groupby(ext["c_Li"].round(3), group_keys=False)
          .apply(lambda x: x.head(MAX_PER_COMP)))          # cap any single composition
ext = ext[
    ["name", "ase_atoms", "energy", "forces", "nat", "c_Li", "fmax", "dft"]]
ext = ext.rename(columns={"dft": "from_prototype"}).sample(frac=1, random_state=0).reset_index(drop=True)
ext.to_pickle(OUT, compression="gzip")

print(f"{OUT}: {len(ext)} structures, {ext.nat.sum()} atoms")
print(f"  prototype deformations {ext.from_prototype.sum()}, generated {(~ext.from_prototype).sum()}")
print(f"  nat {ext.nat.min()}..{ext.nat.max()} (median {ext.nat.median():.0f})")
print(f"  c_Li {ext.c_Li.nunique()} unique values, {ext.c_Li.min():.2f}..{ext.c_Li.max():.2f}")
print(f"  |F|max median {ext.fmax.median():.3f}, max {ext.fmax.max():.2f} eV/A")
print("  c_Li counts:", ext.c_Li.round(3).value_counts().sort_index().to_dict())
