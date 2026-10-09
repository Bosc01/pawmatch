# PawMatch brief

This file holds the project plan and every decision made since the plan was written. It is updated whenever a decision changes. The reasoning behind each decision is in [decisions.md](decisions.md), and data access and protocol details are in [DATA.md](DATA.md).

## Goal

PawMatch is an open-set lost-dog reunification system. Given a photo of a found dog, and optionally a written description, it returns the most likely lost-dog reports, or says there is no match. It is planned as a feature of Stay (trystay.org, React and FastAPI), which helps dog owners avoid surrendering their dogs and is piloting with Austin Pets Alive.

## Standard of evidence

The project matches the standard of the author's earlier work:

- a two-stage retrieval pipeline (FAISS recall, cross-encoder rerank) with a control arm and Benjamini-Hochberg correction across 14 metrics;
- an HNSW index written from scratch in Python (vecstore);
- a C++20 matching engine whose README reports every optimization with before and after numbers, including passes that measured neutral.

## Ground rules

1. Every number comes from a committed `results/*.json` file. README tables are generated from those files by a script, so the README cannot drift.
2. Failures and neutral results are reported next to wins. Nothing is rounded in the flattering direction. A result worse than a published baseline is stated plainly.
3. Splits are by dog identity. No test dog appears in training.
4. Anything synthetic (generated descriptions, simulated locations, padded vectors, simulated corruptions) is labeled synthetic wherever it appears.
5. Hardware is an Apple M5 Air with 24 GB RAM, using PyTorch with MPS. Embeddings are cached, frozen-backbone experiments come first, and there are at most three PetFace fine-tunes. Any subsampling is stated in the README.
6. Work is committed after each step and is never pushed, deleted, or shared without explicit approval.
7. Work stops for a decision only when PetFace data is needed and not present, a PetFace fine-tune is about to start, a choice would rest on a DogFaceNet number, anything would be pushed, deleted, or shared, or Milestone 5 would start.
8. A latency gain is credited only to the optimization pass that produced it, and only if it holds in all three benchmark rounds.

## Milestones

Amendments to the original plan are marked **Amended**.

### Milestone 0: data access (done; waiting for the PetFace download link)

PetFace (ECCV 2024, arXiv 2407.13555): the dog subset has 46,755 individuals and 168,348 images. Access is by request form only, the terms are non-commercial research only, and no images are committed or redistributed. DogFaceNet (CC BY 4.0) is a secondary dataset, used only to build and test the pipeline and for a later cross-dataset evaluation, with nothing tuned on it.

### Milestone 1: embeddings and baselines (pipeline built on DogFaceNet; PetFace pending)

- Identity-disjoint split, frozen and hashed. **Amended:** PetFace uses its official split files. The official validation dogs are halved by identity: val_a calibrates the open-set threshold and chooses the arm for run (c), and val_b trains the Milestone 3 ranker. DogFaceNet uses a seeded 70/10/20 identity split with validation halved the same way.
- Gallery and query protocol on the test dogs: one photo per dog in the gallery, the rest as queries. Open-set variant: 50% of query dogs have no photo in the gallery. Dogs with two photos stay in.
- Frozen baselines: ImageNet ResNet-50, CLIP ViT-B/16, DINOv2 ViT-S/14. Each uses its own normalization; every image is resized to 224 x 224 with no crop.
- Fine-tune DINOv2 ViT-S/14 with ArcFace, with and without batch-hard mining (the hard-negative ablation).
- Metrics: rank-1, rank-5, mAP (leave-one-out over all test photos, since a one-photo gallery makes mAP equal to MRR), and verification AUC, each with a 95% bootstrap interval from 1,000 resamples over query dogs.
- **Amended:** the comparison with the paper uses verification AUC on the official `verification.csv` only (paper: 99.45% for ArcFace on dogs), with both raw AUC and the paper's clamped AUC reported. The paper's 77.86% top-1 measures classification of dogs seen in training and is not compared. The gallery and query results are reported as this project's own protocols.

### Milestone 2: two-stage retrieval and open set (plumbing built on DogFaceNet; PetFace pending)

- Stage 1: FAISS `IndexFlatIP` and `IndexHNSWFlat`, and the vecstore ablation, reported with recall and latency against FAISS.
- Stage 2: rerank of the top 100 by mutual nearest-neighbour matching of DINOv2 patch tokens, with a control arm that uses ImageNet ResNet-50 features. **Amended:** the patch rerank is kept as a reported negative result. No fusion weight is tuned. The small cross-attention reranking head is built after PetFace arrives and is trained on the PetFace train split only.
- Open set: thresholds calibrated on val_a. TAR at FAR 1% and 0.1%, and DIR at the same FARs. **Amended:** the identification threshold solves 1 - F(t)^N = target FAR, where F is the val_a single-impostor score distribution and N is the test gallery size. The achieved test FAR is always reported next to the target.
- Domain shift: a SYNTHETIC degraded query set (motion blur, low light, JPEG compression, partial crop) at one fixed severity each, with the rank-1 drop per corruption. Run (c) tests whether training-time corruption augmentation closes the gap.
- Benjamini-Hochberg correction across every significance test in the project.

### Milestone 3: multimodal learned ranker (waits for PetFace)

- Text: owner-style descriptions generated from PetFace attributes with templates and a small paraphrase step, labeled SYNTHETIC. A text encoder head is aligned with dog embeddings by contrastive loss, and text-to-image recall@10 is evaluated on its own. **Amended:** dogs have breed and sex annotations but no color or pattern annotations, so the main results use breed and sex only. Coat color is an ablation: one label per dog, taken from photos not in the gallery, by a model that is not a retrieval backbone, and labeled MODEL-DERIVED. 200 hand-labeled dogs measure its accuracy, and the ranker is reported with and without color.
- Location and time: last-seen and found locations simulated from a stated movement assumption, labeled SYNTHETIC.
- Learned ranker: LightGBM LambdaMART over stage-1 score, stage-2 score, text similarity, distance, time gap, and breed and color match, trained on val_b. Compared against each single signal and a hand-tuned weighted sum on NDCG@10, MRR, and rank-1, with feature importance and a drop-one-signal ablation.

### Milestone 4: C++ serving (done on DogFaceNet vectors)

- Flat binary export of gallery embeddings, and a C++20 brute-force SIMD scorer (with its own HNSW if time allows).
- GoogleTest: scores match Python within 1e-5 on 10,000 queries.
- p50, p99, and p99.9 latency at gallery sizes 1K, 10K, and 50K, heap allocations counted on the hot path (target zero), and at least two measured optimization passes. **Amended:** DogFaceNet has 7,013 non-query embeddings, so the 10K and 50K galleries are padded with SYNTHETIC random unit vectors. Brute-force latency does not depend on vector values, and the latency results go in `results/` with the hardware named.

### Milestone 5: Stay integration (on hold)

A FastAPI service in this repo, `POST /match`, takes a photo and an optional description and returns the top 5 candidates with scores and a no-match flag. A pull request to Stay, opened but not merged, adds a "Found a dog?" page that uses only a demo gallery built from the test split, stores no user photos, collects no personal data, and shows a visible demo note. **Amended:** this waits until the PetFace authors confirm that their research-only terms allow it.

## Training decisions

- **Budget:** three PetFace fine-tunes. (a) ArcFace, (b) ArcFace with batch-hard mining, (c) the better of the two by closed-set rank-1 on val_a (arm (a) on an exact tie) with corruption augmentation at probability 0.5. Full fine-tunes run on PetFace only; DogFaceNet allows smoke tests of at most 20 steps.
- **Head warm-up**, added after a 5-step smoke test lowered rank-1: the first 10% of steps train the ArcFace head on a frozen backbone, then the backbone unfreezes at learning rate 1e-5 against 1e-3 for the head, and both follow one cosine decay. The values come from standard practice, not from DogFaceNet numbers, and are the same for both arms.
- **Length:** 18,000 steps at batch 64 per PetFace run, about the image count of the paper's 10-epoch recipe, fixed before the first run.
- **Precision:** bf16 autocast for all three PetFace fine-tunes; embeddings are always fp32.

## Engineering

- pytest covers splits, metrics pinned to hand-computed values, the leakage guard, the separation of development results from reported results, and a smoke run on 500 dogs.
- GitHub Actions runs the Python tests and the C++ tests on every push.
- Make targets: `data`, `embed`, `eval`, `train`, `retrieve`, `export`, `cpp-test`, `bench`, `test`, `readme`. `rank` follows with Milestone 3.

## README plan

The README opens with the problem in two sentences, generated from `results/external.json`. **Amended:** the shelter figures come from Shelter Animals Count, 2025 Annual Data Report (released February 4, 2026): 5.8 million dog and cat community intakes and 638,000 returned to owners, both covering dogs and cats together. The microchip claim is sourced to Lord et al., J Am Vet Med Assoc 2009 (peer reviewed, PMID 19601734): microchipped stray dogs were returned to owners 52.2% of the time against 21.9% for the comparison group. The percentages are as reported by JAVMA News in 2017, since the paper's full text was not accessible, and the two secondary sources describe the comparison group differently. A headline results table follows, then sections on data and splits, baselines against the published results, two-stage retrieval, open-set results, domain shift, the multimodal ranker, C++ serving, and known limitations: benchmark photos are not real lost-pet reports, descriptions and locations are synthetic, and there is no live deployment yet.
