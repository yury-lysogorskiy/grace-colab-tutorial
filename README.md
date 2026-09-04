# GRACE Al–Li tutorial

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yury-lysogorskiy/grace-colab-tutorial/blob/main/GRACE-AlLi-tutorial.ipynb)

Finetune the `GRACE-3L-OMAT-large` foundation model on the Al–Li convex hull **plus 128 DFT structures the
foundation model picks itself** from 1000 candidates (`grace_uq select --strategy fps-all`), build
uncertainty artifacts (`grace_uq build`), distil into a fast GRACE/FS student, run LAMMPS MD with an on-the-fly
extrapolation grade, and select the structures worth computing next (`pace_select`). One notebook, twelve
sections, every stage cached.

## Run it

Click the badge, choose a GPU runtime (T4), run all cells. The first cell installs `tensorpotential`,
`amstools` and the GRACE/FS evaluator from `python-ace` and clones this repository; with the shipped
results a first pass takes about 15 minutes. Set `FORCE_RERUN = True` in the configuration cell to
recompute every stage, about an hour on a T4.

Locally: clone, start the notebook from an environment with `tensorpotential>=0.6.1`, `amstools` and
`python-ace` (the wheel from TestPyPI, or the `feature/build-speedup` branch built from source); `ROOT` is the repository directory.

## What is in the repository

| path | content |
|---|---|
| `GRACE-AlLi-tutorial.ipynb` | the tutorial |
| `0-data/` | DFT inputs: 112 + 16 hull structures, 32 OOD structures of an unseen prototype, 265 evaluation structures, 1000 selection candidates, 19 Materials Project cells to relax, reference numbers of the hull-only run, and the scripts that prepared the sets |
| `1-select/` | foundation-model features of the candidates and the 128 selected structures |
| `1-finetune-hull+fps/` | the finetuned teacher as its UQ SavedModel (energies, forces, stress and γ), the UQ artifact, metrics |
| `2-uq-validation/`, `3-distill/`, `4-convex-hull/`, `5-lammps-gamma-le-5/` | gamma caches, distillation pool and labels, the GRACE/FS student with its active set, relaxed structures, the MD log and the extrapolative frames |
| `docs/known-issues.md` | tool quirks that the notebook works around |

The teacher's training checkpoints are not shipped (120 MB); with `FORCE_RERUN` the teacher is retrained.

## LAMMPS

Section 12 re-runs the MD only if a LAMMPS binary with GRACE/FS and Kokkos/CUDA is present. On Colab,
put the Google Drive id of `lmp-grace-t4-cu128.tar.gz` into `LMP_DRIVE_ID` in the setup cell. Without
it the section shows the shipped run.

## Provenance

DFT reference data: the Al–Li database of S. Menon, Y. Lysogorskiy, A. L. M. Knoll, N. Leimeroth, M. Poul,
M. Qamar, J. Janssen, M. Mrovec, J. Rohrer, K. Albe, J. Behler, R. Drautz and J. Neugebauer, *From electrons to
phase diagrams with machine learning potentials using pyiron based automated workflows*, npj Comput. Mater. **10**,
261 (2024), [doi:10.1038/s41524-024-01441-0](https://www.nature.com/articles/s41524-024-01441-0). Please cite it
when you use these data. The reference numbers in `0-data/reference-hull-only-*`
come from the same pipeline trained on the 128 hull structures only, student on the full distillation
pool, 2026-09-03.
