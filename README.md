# GRACE Al–Li tutorial

<a href="https://colab.research.google.com/github/yury-lysogorskiy/grace-colab-tutorial/blob/main/GRACE-AlLi-tutorial.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>

📖 **Documentation: [gracemaker.readthedocs.io](https://gracemaker.readthedocs.io/)**

Finetune the `GRACE-3L-OMAT-large` foundation model on the Al–Li convex hull **plus 128 DFT structures the
foundation model picks itself** from 1000 candidates (`grace_uq select --strategy fps-all`), build
uncertainty artifacts (`grace_uq build`), distil into a fast GRACE/FS student, validate all four models
(energy–volume curves, elastic constants, phonons with `amstools`), run the same LAMMPS heating ramp with the
student, the finetuned teacher and the untouched foundation model, each reporting its own extrapolation
grade γ on the fly, and select the structures worth computing next (`pace_select`). One notebook, thirteen
sections and an appendix with every shell command, every stage cached.

## Run it

Click the badge, keep the preselected **T4 GPU** runtime, run all cells. The setup cell installs
`tensorpotential`, `amstools` and `python-ace`, downloads the two foundation models with their checkpoints
(`grace_models download`, `grace_models checkpoint`), a LAMMPS binary built for the T4 and the Kokkos export
of the finetuned teacher (both from this repository's releases), and clones this repository with its cached
results. A first pass over the shipped results takes about 15 minutes. Set `FORCE_RERUN = True` in the
configuration cell to recompute every stage, about an hour on a T4. Without a GPU the cached pass still
runs, only slower where a model is evaluated live.

Locally: clone, start the notebook from an environment with `tensorpotential>=0.6.1`, `amstools`,
`phonopy<4` and `python-ace>=0.4.0rc1`; `ROOT` is the repository directory and `LMP` in the configuration
cell should point to your own Kokkos/CUDA build of LAMMPS with the ML-PACE package.

The notebook in the repository carries the outputs of a cached run, so the expected results are visible
before anything is executed. To keep them after editing in Colab, use *File → Save a copy in GitHub* on this
repository.

## What is in the repository

| path | content |
|---|---|
| `GRACE-AlLi-tutorial.ipynb` | the tutorial, with outputs |
| `figures/` | the pipeline scheme shown under the title, with the script that draws it |
| `0-data/` | DFT inputs: 112 + 16 hull structures, 32 OOD structures of an unseen prototype, 265 evaluation structures, 1000 selection candidates, 19 Materials Project cells to relax, and the scripts that prepared the sets |
| `1-select/` | foundation-model features of the candidates and the 128 selected structures |
| `1-foundation-baseline/` | energies of the dataset from `GRACE-3L-OMAT-large` and `GRACE-FS-OMAT` before any finetuning |
| `1-finetune-hull+fps/`, `train-hull+fps.pkl.gz` | the finetuned teacher as its UQ SavedModel (energies, forces, stress and γ), the UQ artifact, metrics, and the training set it was fitted to |
| `2-uq-validation/`, `3-distill/`, `4-convex-hull/` | γ caches, distillation pool and labels, the GRACE/FS student with its active set, relaxed structures |
| `6-validation/` | energy–volume, elastic and phonon results of the four models for fcc Al and B32 AlLi |
| `5-lammps-student-fs-gamma-le-5/`, `5-lammps-teacher-3L-rp128_p99/`, `5-lammps-foundation-3L-OMAT/` | the three MD runs: data file, log, extrapolative frames, and the student's `pace_select` result |
| `docs/known-issues.md` | tool quirks that the notebook works around |

Not in git, fetched by the setup cell from the
[release](https://github.com/yury-lysogorskiy/grace-colab-tutorial/releases/tag/lammps-t4-v1): the LAMMPS
binary and the teacher's Kokkos export `kokkos_uq_rp128_p99.npz` (74 MB). The teacher's training checkpoint
is not needed for the cached pass; with `FORCE_RERUN` the teacher is retrained, and the checkpoint is also
available on the same release as `teacher-checkpoint_best_test_loss.tar.gz`.

## LAMMPS

Section 13 drives the same 500-step heating ramp, 500 → 5000 K under NPT, with three potentials:

| run | pair style | model file | γ |
|---|---|---|---|
| student | `grace/fs/kk` | `saved_model.yaml` + active set `.asi` | D-optimality (MaxVol) |
| finetuned teacher | `grace/3l/kk` | `kokkos_uq_rp128_p99.npz` from `grace_utils export_kokkos --uq-artifacts` | NCM (nearest-cluster Mahalanobis distance) of the UQ artifact |
| foundation model | `grace` (TensorFlow) | the `GRACE-3L-OMAT-large` SavedModel | NCM shipped with the model |

The runs are re-executed only when their log is missing or `FORCE_RERUN` is set; otherwise the shipped logs are
plotted. The release provides two binaries compiled for the T4 (compute capability 7.5, Kokkos/CUDA, no MPI):
`lmp-grace-t4-tf.tar.gz`, the default, whose `grace` styles use Colab's own TensorFlow, and `lmp-grace-t4.tar.gz`
without any TensorFlow dependency (`grace/fs`, `grace/*/kk` and `pace` only). Elsewhere point `LMP` in the
configuration cell to your own build.

## Provenance

DFT reference data: the Al–Li database of S. Menon, Y. Lysogorskiy, A. L. M. Knoll, N. Leimeroth, M. Poul,
M. Qamar, J. Janssen, M. Mrovec, J. Rohrer, K. Albe, J. Behler, R. Drautz and J. Neugebauer, *From electrons to
phase diagrams with machine learning potentials using pyiron based automated workflows*, npj Comput. Mater. **10**,
261 (2024), [doi:10.1038/s41524-024-01441-0](https://www.nature.com/articles/s41524-024-01441-0). Please cite it
when you use these data.
