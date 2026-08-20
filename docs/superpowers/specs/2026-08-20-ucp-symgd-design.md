# UCP/SymGD Integration Design

## Goal

Add an opt-in Unified Copy-Paste (UCP) and Symmetric Guidance (SymGD)
training branch to SemiSAM-O1 for the MT and UAMT backbones. The branch starts
at Round 2 by default, leaves the original training and KNN refinement paths
unchanged when disabled, and does not invoke SAM during online training.

## Scope

The first implementation supports `mt` and `uamt`. DAN and DTC are excluded
because they do not currently own EMA teacher models; adding teachers to those
backbones would change their algorithms and memory profiles.

The change covers training-time UCP and SymGD only. It does not feed voxel-wise
XOR masks into KNN refinement. SemiSAM-O1 ranks uncertainty per volume using
mean voxel entropy and replaces uncertain volumes through global-feature KNN
voting, whereas SymGD uses disagreement as a voxel-wise training mask.

## Command-Line Interface

The following arguments are added to `code/train_SemiSAM_O1.py`:

- `--ucp_symgd`: opt-in boolean flag; disabled by default.
- `--ucp_start_round`: first round that enables the branch; default `2`.
- `--ucp_scale_min`: minimum side-length ratio for the centered cuboid; default
  `0.3`.
- `--ucp_scale_max`: maximum side-length ratio for the centered cuboid; default
  `0.6`.
- `--symgd_confidence`: minimum teacher confidence for direct and merged views;
  default `0.95`.
- `--symgd_weight`: maximum symmetric-guidance loss weight; default `1.0`.

Startup validation requires:

- `ucp_start_round >= 1`;
- `0 < ucp_scale_min <= ucp_scale_max <= 1`;
- `0 <= symgd_confidence <= 1`;
- `symgd_weight >= 0`;
- `--ucp_symgd` may only be used with `mt` or `uamt`.

Invalid configurations fail before loading data or models. The ordinary MT,
UAMT, DAN, and DTC computational paths remain functionally unchanged when the
flag is absent.

## Components

### Pure UCP/SymGD Utilities

Create `code/utils/ucp_symgd.py` with small tensor functions that run on CPU or
CUDA:

1. Build one centered 3D cuboid mask per unlabeled sample. Each side-length
   ratio is sampled uniformly from the configured range. The returned mask has
   shape `[N_u, 1, D, H, W]` and the same device/dtype as the input images.
2. Pair every unlabeled sample with a labeled sample by cycling labeled batch
   indices. This supports the default `1 + 1` batch and avoids a special-case
   failure when batch sizes change.
3. Produce inward and outward images and labels:

   ```text
   U_in  = X_l * M       + X_u * (1 - M)
   Q_in  = Y_l * M       + Y_u * (1 - M)
   U_out = X_u * M       + X_l * (1 - M)
   Q_out = Y_u * M       + Y_l * (1 - M)
   ```

4. Reconstruct the unlabeled teacher view from the two mixed predictions:

   ```text
   P_merged = P_out * M + P_in * (1 - M)
   ```

5. Build a symmetric guidance mask. A voxel is retained only when the direct
   and merged teacher argmax labels agree and both maximum softmax
   probabilities meet `symgd_confidence`.
6. Compute masked cross entropy from the direct student logits to the detached
   merged teacher hard labels. If the mask is empty, return a differentiable
   zero rather than dividing by zero.

The utilities contain no model construction, data loading, global state, or
new dependency.

### Shared Training Branch

Add one shared helper in `code/train_SemiSAM_O1.py` for MT and UAMT. It receives
the student, EMA teacher, current batch, existing CE/Dice losses, iteration,
and round number. When the feature is inactive it returns zero losses without
running extra forwards.

When active, it:

1. Splits the already augmented batch according to `labeled_bs`.
2. Builds inward/outward UCP images and hard targets from the real label and
   current round's refined pseudo-label.
3. Runs one concatenated student forward over both mixed batches.
4. Runs one detached concatenated EMA teacher forward over both mixed batches.
5. Computes the mean CE+Dice loss across inward and outward student outputs.
6. Reconstructs the merged unlabeled teacher view and computes the confidence
   and agreement filtered SymGD loss against the existing direct unlabeled
   student output.
7. Returns UCP loss, SymGD loss, and the retained-voxel ratio for logging.

The direct teacher prediction already computed by MT/UAMT is reused. No extra
direct teacher forward is added.

## Loss Integration

The existing backbone loss remains intact. For an active branch:

```text
L_total = L_existing
        + pseudo_weight * L_ucp
        + gamma(t) * L_sym
```

`pseudo_weight` is the existing linear ramp from 0 to 1 during the first 30%
of a round. The SymGD multiplier follows the approved linear schedule:

```text
gamma(t) = symgd_weight * (0.1 + 0.9 * clamp(t / max_iterations, 0, 1))
```

Teacher probabilities and merged hard labels are detached. Gradients flow
through the direct student prediction for `L_sym` and through the two mixed
student predictions for `L_ucp` only.

## Data Flow and Invariants

The existing `TwoStreamBatchSampler` keeps labeled samples before unlabeled
samples. UCP operates after the existing paired random rotation, flip, crop,
and tensor conversion, so every mixed image and label has the same spatial
shape. It does not alter the dataset, sampler, pseudo-label dictionaries, SAM
features, validation, checkpoint format, resume behavior, or between-round KNN
refinement.

The implementation remains multiclass: hard labels use integer class IDs,
teacher decisions use softmax/argmax, and cross entropy is used instead of a
binary-only BCE formulation.

## Logging

For active rounds, log these values alongside the existing losses:

- UCP CE+Dice loss;
- SymGD masked cross-entropy loss;
- SymGD retained-voxel ratio;
- current SymGD weight.

TensorBoard scalar names use a `train/` prefix and the text log adds the new
values to the existing 100-iteration message. Disabled and Round 1 runs do not
emit misleading active-branch metrics.

## Error Handling

Invalid CLI ranges or unsupported backbones raise `ValueError` before CUDA,
checkpoint, or data access. Runtime utilities validate tensor ranks, compatible
spatial shapes, nonempty labeled/unlabeled partitions, and class-logit shapes.
Errors name the violated invariant.

An empty confidence/agreement mask is a valid training state. Its symmetric
loss and retained ratio are both zero.

## Verification

Create `code/tests/test_ucp_symgd.py` using Python's standard `unittest`
runner. CPU tests cover:

- centered cuboid geometry and scale bounds;
- inward/outward image and label mixing;
- cycling labeled samples when batch counts differ;
- exact reconstruction of the unlabeled mixed teacher view;
- exclusion of disagreement and low-confidence voxels;
- finite differentiable zero loss for an empty mask;
- multiclass masked cross-entropy and gradient flow.

Run:

```powershell
python -m unittest discover -s code/tests -p "test_*.py" -v
python -m compileall -q code
```

The available machine has CPU-only PyTorch and no HDF5 training volumes, so a
CUDA training run is outside local verification. The usage guide provides a
short MT/UAMT smoke-run command for the user to execute in the target training
environment.

## Documentation

Create `docs/ucp_symgd_usage.md` with:

- baseline and enabled MT/UAMT commands;
- all new arguments and defaults;
- explanation that the branch starts at Round 2 unless overridden;
- expected additional student/teacher forwards and VRAM cost;
- TensorBoard metrics to inspect;
- CPU validation commands and a short CUDA smoke-run procedure;
- an explicit note that KNN refinement remains sample-level and unchanged.

## Sources

- SemiSAM-O1 paper: <https://arxiv.org/html/2604.24109>
- SymGD paper: <https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Constructing_and_Exploring_Intermediate_Domains_in_Mixed_Domain_Semi-supervised_Medical_CVPR_2024_paper.html>
- SymGD reference implementation: <https://github.com/MQinghe/MiDSS>
