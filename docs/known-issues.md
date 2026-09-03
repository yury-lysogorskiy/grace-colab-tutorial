# Known issues and tool quirks

Developer notes that were removed from the tutorial notebook so that students see the workflow,
not the workarounds. Keep this list in sync with the tools; delete an entry once the tool is fixed.

## Jupyter

- `!cmd` in a notebook does not surface the exit status of the command reliably, and the
  console scripts of the kernel's environment (and its `libstdc++`) are not on the shell's
  `PATH`/`LD_LIBRARY_PATH` unless the kernel puts them there. The notebook therefore runs every
  tool through a small `run()` helper built on `subprocess` (`check`-style, streams output,
  inherits `os.environ`) instead of `!`. The helper starts the command in its own process group
  and forwards a cell interrupt (the stop button) to that group, so stopping a cell also stops
  `gracemaker` or LAMMPS instead of leaving it running on the GPU. Colab runs `subprocess` exactly
  as Jupyter does; `!` is not required there.

## gracemaker

- `trainable_variable_names` for frozen-readout finetuning is **tier specific**. The 3-layer
  list (`rho1/reducing_ ... eq2/reducing_`) differs from the 2-layer and 1-layer ones. A wrong
  list trains nothing and fails silently.
- `train_metrics.yaml` / `test_metrics.yaml` are **appended to**, not overwritten. A re-run in the
  same directory, or any step that re-enters the fit machinery, leaves several blocks in one file
  with `epoch` restarting at 0. Reading `.iloc[-1]` then reports the wrong row. The current
  artifacts have single blocks; if a learning curve ever shows the epoch counter jumping back to
  zero, this is why.
- `gracemaker -r -sf` (resume + export FS yaml) fails on a **finetuned** FS model: `shift: auto`
  needs the foundation-model checkpoint, which the resume path does not carry. Use
  `grace_utils -p model.yaml -c <checkpoint> export -sf` instead.
- GRACE/FS has no frozen-readout mode (its readout block is `E`, not `rho`), so FS finetuning is
  always a full-parameter finetune.

## grace_uq

- `grace_uq build` **reuses `.step1_final.npz` / `.step2_final.npz` checkpoints** left in the
  working directory by a previous build. If they came from a run with different settings (another
  `--rp-dim` or cluster count) the step-3 worker dies with an opaque `worker(s) [0] failed` and no
  hint that stale state is the cause. Always pass `--restart`.
- `--threshold-percentile` (percentile calibration of gamma) is newer than some released versions.
  Without it the artifact uses the median + 3 MAD outlier fence, which on this dataset puts
  about 16 % of the *training* atoms above gamma = 1 instead of about 1 %. The artifact and
  SavedModel names in the notebook carry the calibration (`_p99`) so the two are never confused.

## grace_predict

- The output dataframe contains `energy_predicted` / `forces_predicted` keyed by `name` and **no
  `ase_atoms`**. It must be merged back onto the input structures before it can be used as a
  training set.

## pace_activeset

- Occasionally dies during MaxVol without a traceback. With `run()` this surfaces as a non-zero
  exit code; re-run the cell.

## amstools.thermodynamics

- `ensure_energy_per_atom_column`, `compute_compositions`, `compute_formation_energy` and
  `compute_convexhull_dist` **mutate the dataframe in place** and return `None` or the element
  list, never the dataframe. Assigning their result silently nulls the dataframe.
- They also **skip a column that already exists**, so a dataframe that carries DFT-derived
  `energy_per_atom` / `e_formation_per_atom` / `e_chull_dist_per_atom` must have those columns
  dropped before recomputing them from model energies. The notebook's `with_hull()` does this.
- `run_convex_hull_calculation` returns `(dataframe, pipeline_dict)` although its annotation says
  `DataFrame`, and the returned `ase_atoms` carry live calculators that are not picklable; copy the
  atoms (`Atoms.copy()` drops the calculator) before saving.

## LAMMPS

- Kokkos **OpenMP** threading is markedly slower than plain MPI ranks for `grace/fs` on CPU:
  measured on two cores, about 2.3 katom-step/s versus about 7.9 for two MPI ranks. The CPU path
  therefore uses `mpirun -np N` with `OMP_NUM_THREADS=1`.
- Kokkos/CUDA builds are architecture specific; `GPU_BUILDS` in the notebook's configuration maps
  the GPU name reported by `nvidia-smi` to a build directory.

## Data and references

- **Names in the source database are not unique.** `AlLi_BIG_Oct2023_UNIQUE_free-e.pkl.gz` reuses
  `structure_N` names across LADDER generations (41 repeats inside a random draw of 1000). Two
  consequences: (1) anything keyed on `name` (`grace_predict`, `grace_uq select`, dataframe merges)
  needs unique ids, which is why `make_candidates.py` renames candidates to `cand_0000 ...` and keeps
  the original in `source_name`; (2) **a filter that re-selects rows by name lets the twins of an
  accepted structure through.** `make_candidates.py` had this bug on 2026-09-03 (candidates up to
  3.5 eV/atom above the hull despite a 500 meV/atom filter) and now selects by row.
  `make_extended_eval.py` still uses the name-based hull filter: the shipped
  `AlLi-extended-eval.pkl.gz` has 68 of 265 structures above 500 meV/atom (max 2.35 eV/atom). Its
  error table is therefore harder than its description says; regenerating it changes every number
  measured on it, including the hull-only reference files.

- Materials Project and the tutorial's VASP dataset differ by about 58 meV/atom in the Al–Li hull
  minimum (MP at c_Li = 0.5, this data at c_Li = 0.6). Errors and the gamma-vs-error test must be
  measured against the reference the model was fitted to; against MP there is no correlation.
- The 19 MP candidate structures used in §10 are shipped as `0-data/MP-hull-candidates.pkl.gz`
  (originally fetched with `amstools.sources.fetch_structures(["Al", "Li"], max_atoms=32)`), so no
  MP API key or network access is needed.
