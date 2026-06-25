# Deep Analysis — Paper-Log Papers 2, 4, 6

Extensive read of the three papers flagged in the log, and how each modifies our plan.
**Note:** only paper 4 actually uses TN5000; papers 2 and 6 use other data/tasks.

---

## Paper 2 — Attention-mechanism comparison (Sci Reports 2024)
*"Identification of lesion location and discrimination between benign and malignant…"*
[PMC11685495](https://pmc.ncbi.nlm.nih.gov/articles/PMC11685495/)

- **Task/data:** YOLOv8 **object detection** on thyroid ultrasound **(grayscale + elastography)** — *not* TN5000, *not* a classification task. Metric is **mAP@50**, not accuracy.
- **Attention comparison (the key bit):** **CPCA best (91.5% mAP@50)**, **CBAM 2nd (90.8%)**,
  Coordinate-Attention / STN added only +0.2–0.4%. Final diagnostic rates 89.3% benign / 90.4% malignant.
- **Their own conclusion:** attention is **architecture- and task-specific**; *"CBAM underperformed here"* and the choice **must be tested, not assumed**.
- **⚠️ Correction to a common misreading:** this paper does **not** show "CBAM is great / the best." CBAM was a strong 2nd in a *detection* task; **CPCA won**.
- **Implication for us:** *strengthens* the plan to run a controlled **baseline / +SE / +CBAM** comparison (don't assume CBAM); **CPCA is a candidate 4th arm** worth noting. It gives us **no CBAM number to cite as a win.**

## Paper 4 — TN5000 dataset paper (Nature Sci Data 2025)
[s41597-025-05757-4](https://www.nature.com/articles/s41597-025-05757-4)
- 5,000 biopsy-confirmed B-mode images; **official image-level random 7:1:2 split**, ships detection + classification benchmark baselines.
- **Confirms the root cause of our duplicate-leakage finding** (image-level random split, no de-dup). This is our primary training set.
- **Implication:** corroborates the duplicate finding; their baseline numbers are the in-dataset reference.

## Paper 6 — Segmentation→classification + cross-dataset (arXiv 2025)
*"A Deep Learning Framework for Thyroid Nodule Segmentation and Malignancy…"*
[arXiv 2511.11937](https://arxiv.org/abs/2511.11937)
- **Pipeline:** TransUNet **segmentation** → **ResNet-18** classifier.
- **Data:** segmentation trained on **DDTI+TN3K**, externally tested on **TNUI**. **Classification** = 5-fold CV on **DDTI only** (349 images, 61 benign / 288 malignant), **F1 0.852 / acc 0.782**; RF baseline F1 0.829.
- **⚠️ Key catch:** their *classification* was **NOT** externally validated — only **segmentation** crossed datasets.
- **Implication for us:** our planned **TN5000→DDTI classification** cross-test is **more rigorous** than theirs (they never cross-validated classification across institutions). When we test on DDTI, note it is small + heavily malignant-skewed → **lead with sensitivity/specificity**, not accuracy.

---

## How this modifies the plan
1. **Attention (Phase 2):** run **baseline / +SE / +CBAM** (and consider **CPCA**) as a tested comparison — paper 2 proves attention choice is task-specific and CBAM is not automatically best.
2. **Cross-dataset (Phase 3):** our TN5000→DDTI classification test is a genuine step beyond paper 6; design DDTI eval for class imbalance (sens/spec, not accuracy).
3. **Backbone:** the thyroid-US literature + our bake-off both favour ResNet-50; tonight's two-phase re-train settles the baseline pick.
