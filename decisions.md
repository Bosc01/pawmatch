# Decisions

Data, split, and evaluation-protocol decisions are in [DATA.md](DATA.md). This file records training decisions and changes to the original project plan. Numbers below that come from `results/dev/` are DogFaceNet development measurements, not results.

## Fine-tune budget (2026-10-08)

Three PetFace fine-tunes, replacing the original limit of two:

- (a) `arcface`: ArcFace, no hard-negative mining.
- (b) `arcface_bh`: ArcFace plus batch-hard triplet loss.
- (c) `augmented`: the better of (a) and (b), plus the Milestone 2 corruption augmentation.

(a) against (b) is the hard-negative ablation. Both use the same sampler, schedule, seed, and warm-up, so the loss term is the only difference. Head warm-up is part of each run and does not count against the budget. DogFaceNet smoke runs (at most 20 steps, enforced in code) do not count either. The code refuses to repeat a PetFace run whose checkpoint already exists.

Pending confirmation: "better" is defined as higher closed-set rank-1 on val_a, so that run (c) is not chosen on test data.

## Head warm-up (added 2026-10-08, after the 5-step smoke test)

Why: in the 5-step smoke test on DogFaceNet (commit 41db12f), fine-tuning lowered DINOv2 rank-1 from 77.93% (frozen) to 71.04% for arm (a) and 74.89% for arm (b). Five steps cannot separate the arms, but the drop matches a known effect: gradients from a randomly initialized head distort pretrained features early in fine-tuning. Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution" (ICLR 2022), propose training the head first (LP-FT).

What:

- The first 10% of steps train only the ArcFace head, with the backbone frozen, at head learning rate 1e-3.
- The backbone then unfreezes at learning rate 1e-5, 100 times below the head (discriminative learning rates, Howard and Ruder, ULMFiT, ACL 2018).
- Both parameter groups then follow one cosine decay to zero.

How the values were fixed: 10% is a conventional warm-up fraction and was not tuned. Both learning rates are unchanged from the values set before the smoke test. No value was chosen from a DogFaceNet number, and both arms use the same warm-up.

## Training length (2026-10-08)

Each PetFace run is 18,000 steps at batch 64 (32 dogs x 2 photos), of which the first 1,800 are head warm-up. The length is fixed before the first run and does not change when the real `train.csv` count is known.

Basis: the PetFace ArcFace recipe trains 10 epochs (`src/configs/dog.py`). Taking 70% of 168,348 dog photos, about 117,844, as the training set, 10 epochs is about 1.18 million image presentations, or about 18,400 steps at batch 64. This was rounded down to 18,000.

## Mixed precision (2026-10-08)

Speed test only, on DogFaceNet: 20 steps of arm (b) in fp32 and in bf16 autocast from the same seed, on an Apple M5 with 24 GB RAM (`results/dev/precision_test.json`). The first step of each phase is excluded from the medians.

| | fp32 | bf16 |
|---|---|---|
| Median seconds per full step | 2.68 | 1.70 |
| Median seconds per head-only step (one sample) | 0.68 | 0.25 |
| Estimated hours per 18,000-step PetFace run | 12.4 | 7.8 |

The largest absolute difference between the two ArcFace loss curves over the 20 steps was 0.0131.

bf16 was tested instead of fp16 because it needs no gradient scaler. The losses stay in fp32 in both modes.

Decision (2026-10-08): all three PetFace fine-tunes train in bf16 autocast (`configs/petface.toml`). Embeddings are always computed in fp32.

## Preprocessing check (2026-10-08)

Each backbone uses the mean, standard deviation, and interpolation from its own timm pretrained config, and `tests/test_embed.py` pins them: ImageNet constants for ResNet-50 and DINOv2, OpenAI CLIP constants for CLIP.

Two settings deliberately differ from the timm eval defaults:

- No center crop, since both datasets are already face crops. timm would crop to 87.5% for ResNet-50 and 90% for CLIP.
- DINOv2 runs at 224 x 224 instead of 518 x 518.

The timm CLIP model was checked against the open_clip reference implementation on 64 DogFaceNet images (`results/dev/clip_reference_check.json`), and the embeddings agree to a minimum cosine similarity of 0.9999998. The official OpenAI preprocessing does not crop a square 224 image, so the CLIP input matches the reference exactly. Nothing changed, and no cache was rebuilt.

## Milestone 2 parameters (2026-10-08)

All fixed before the first Milestone 2 run and not tuned on any data:

- Stage 1: DINOv2 global embeddings. FAISS `IndexFlatIP` (exact) and `IndexHNSWFlat` with M = 32 and efConstruction = 40. efSearch is reported as a sweep over 16, 32, 64, 128, and 256. Only efSearch = 64 enters the significance tests.
- Stage 2 reranks the top 100. The score is the sum of similarities over mutual nearest-neighbour pairs of local features: DINOv2 patch tokens, or for the control arm, ImageNet ResNet-50 final feature-map cells. There are two fusions: the stage-2 score alone, and reciprocal rank fusion with the stage-1 order at k = 60 (Cormack et al., SIGIR 2009).
- Open set: thresholds come from val_a and are applied unchanged to test. TAR uses all pairs of distinct photos. DIR uses the open-set gallery protocol.
- Corruptions (SYNTHETIC, queries only, one severity each):
  - motion blur, 15 px in a random direction;
  - low light: brightness times 0.3, gamma 1.5, Gaussian noise std 0.03;
  - JPEG quality 15 (ImageNet-C severity 3);
  - a random crop keeping 60% of the area.
- Run (c) augmentation: each training photo gets one random corruption from that set with probability 0.5.
- Significance: a two-sided paired sign-flip permutation test with dogs as units, 10,000 permutations, on per-query rank-1. Benjamini-Hochberg runs across every test in `pawmatch/significance.py`.
- The vecstore ablation did not run: github.com/Bosc01/vecstore at commit ac7603b has no `pyproject.toml` or `setup.py`, so pip cannot install it.

## Milestone 4 setup (2026-10-08)

- Gallery padding: there are only 7,013 DogFaceNet DINOv2 embeddings outside the test queries (the test gallery plus every validation and training photo). The 10K and 50K galleries are filled past that with random unit vectors, labeled synthetic in `results/m4_latency.json`. Brute-force latency does not depend on vector values. The queries are the 1,350 test-query embeddings, padded the same way to 10,000.
- Parity: every query is scored against the first 1,000 gallery rows and compared with float64 scores computed in Python. The tolerance is 1e-5 absolute.
- Benchmark: top 100, one thread, one query at a time, 1,000 warm-up queries, then 10,000 timed queries. There are three rounds, each running every size and pass once, and the reported figure is the median across rounds.
- Heap allocations are counted by replacing the global `operator new` in the test and benchmark binaries.
