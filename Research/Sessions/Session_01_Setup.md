# ThyroVision — Session Brief

**Session #:** 01  **Date:** 2026-06-18  **Week of plan:** Week 1–2 (Phase 0→1 setup)
**Stage:** Setup

## 1. Goal for this session
Inspect the two datasets, confirm the machine can train, and stand up the project
skeleton (folders, git, virtual environment, dependencies) — no model or
data-loading code yet.

## 2. What actually happened
- Read all project context: `CLAUDE.md`, `research/Project_Context.md`, the vision
  PDF, and `Architecture.docx`. (Context files were initially empty on disk and had
  to be re-saved before they could be read.)
- **Inspected both datasets (read-only, nothing modified):**
  - **TN5000** — 5,000 JPEG images + 5,000 Pascal-VOC XML annotations under
    `Data/TN5000 Dataset/TN5000_forReview/`, plus the authors' mmdetection benchmark
    code (ignored). Labels are bounding boxes with class `0`/`1`; the split is
    pre-defined (train 3,500 / val 500 / test 1,000). Class balance 1,426 vs 3,574
    (malignant-skewed, matches the doc). **No patient-ID field** — only numeric image
    IDs and the annotator name. Patient-level splitting will not be possible on
    TN5000 from this metadata, and that will be stated explicitly.
  - **DDTI** — 480 JPEGs + 390 case XMLs (one per case; multiple images per case).
    Labels are TI-RADS category + freehand segmentation polygons, with patient/case
    metadata (age, sex, TI-RADS filled in ~75–78% of cases). Case number serves as a
    patient ID. **Left quarantined per rule 1.**
- **GPU check:** no NVIDIA/CUDA. Machine is Apple M5, 16 GB unified memory, Metal 4.
  Accelerator is the Apple GPU via PyTorch's **MPS** backend.
- **Built the project skeleton:** created `src/`, `outputs/{checkpoints,figures,logs}/`,
  `research/Sessions/`; initialised git (branch renamed `master`→`main`); fixed the
  `.gitignore` (was empty) to exclude `Data/`, checkpoints, logs, weights, venv —
  `outputs/figures/` left tracked on purpose.
- **Environment:** created a Python **3.11.15** venv (system default is 3.14, too new
  for current DL wheels). Installed the Phase-1 stack: torch 2.12.1, torchvision
  0.27.1, timm 1.0.27, numpy, pillow, opencv, albumentations, scikit-learn, pandas,
  matplotlib, tqdm. Pinned exact versions to `requirements.lock.txt`.
- **MPS smoke test:** ResNet-50 forward+backward ran on the Apple GPU successfully
  (`MPS available: True`, loss 0.71, ~5 s incl. first-run kernel compile).
- Adopted the folder structure from `Project_Context.md` §5 (chosen over an
  earlier, more elaborate proposal — kept lean to match the project's scope
  discipline). Updated CLAUDE.md dataset paths to the real uppercase folder names.

## 3. Results / numbers
Accuracy: n/a  Sensitivity: n/a  Specificity: n/a  AUC: n/a
(No model trained this session — setup only. The MPS smoke test reported a loss of
0.71 on random data, which only confirms the training path works, not any result.)

## 4. Sanity checks confirmed
- [ ] Split was patient-level (or it was noted explicitly why not) — *N/A this session; noted that TN5000 has no patient ID*
- [ ] Test set stayed sealed until final evaluation — *N/A this session*
- [x] `Data/DDTI Dataset/` was NOT used in training — *only read-only inspection; quarantine intact*
- [ ] Training took realistic time (not instant) — *N/A; no training yet*
- [ ] Reported more than just accuracy — *N/A; no results yet*

## 5. What went wrong / surprised us
- The four context files (`CLAUDE.md`, `Project_Context.md`, `Session_Template.md`,
  `README.md`) were all 0 bytes at first — unsaved in the editor. Resolved by
  re-saving; `README.md` is still empty (a fill-later target, not context).
- System Python is 3.14.4, too new for stable DL wheels — used Homebrew's 3.11.15
  for the venv instead of the planned 3.12 (3.12 wasn't installed; 3.11 is
  well-supported).
- TN5000 having **no patient-ID metadata** is the notable finding — it constrains
  rule 2 (patient-level splits) and must be disclosed in any write-up.
- The dependency install was interrupted once (network); re-ran cleanly.

## 6. Next session
Build the TN5000 data loader: parse the VOC XML, load images + binary labels,
set up the train/val/test split (using the dataset's predefined split, with the
no-patient-ID limitation documented), and verify a few samples visually.

## 7. One line for the mentor
"Project is scaffolded and the Mac is confirmed training-capable on its Apple GPU;
datasets are inspected — TN5000 is bounding-box VOC with no patient IDs, DDTI stays
held out — and we're ready to build the TN5000 baseline loader next."
