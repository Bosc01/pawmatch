# Decisions

Data, split, and evaluation-protocol decisions are in [DATA.md](DATA.md). This file records training decisions and changes to the original project plan. Numbers below that come from `results/dev/` are DogFaceNet development measurements, not results.

## Fine-tune budget (2026-10-08)

Three PetFace fine-tunes, replacing the original limit of two:

- (a) `arcface`: ArcFace, no hard-negative mining.
- (b) `arcface_bh`: ArcFace plus batch-hard triplet loss.
- (c) `augmented`: the better of (a) and (b), plus the Milestone 2 corruption augmentation.

(a) against (b) is the hard-negative ablation. Both use the same sampler, schedule, seed, and warm-up, so the loss term is the only difference. Head warm-up is part of each run and does not count against the budget. DogFaceNet smoke runs (at most 20 steps, enforced in code) do not count either. The code refuses to repeat a PetFace run whose checkpoint already exists.

Rule for run (c), accepted 2026-10-08: the arm for run (c) is whichever of (a) and (b) has the higher closed-set rank-1 on val_a. The measurement uses the Milestone 1 gallery and query protocol with the same seed, so run (c) is never chosen on test data. On an exact tie, arm (a) is used, as the simpler loss. The choice and both val_a numbers are recorded in `results/` before run (c) starts.

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

Decision (2026-10-08): keep the resize to 224 x 224 with no crop for every backbone.

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

## Patch rerank outcome (2026-10-08)

The mutual nearest-neighbour patch rerank stays in the project as a reported negative result. On DogFaceNet development data it lowered rank-1 below stage 1 under both fusions, and the ImageNet control arm was lower still (`results/dev/m2_rerank.json`).

How the stage-1 and stage-2 fusion was chosen: no weight was tuned. One fusion ranks by the stage-2 score alone. The other is reciprocal rank fusion, which weights the two ranks equally with k = 60 from Cormack et al. (SIGIR 2009). Both were in the code before the first run (commit bb653c2 introduces `RRF_K = 60`, and no later commit changes it). No split, test or otherwise, was used to choose them, so nothing needed to be refit on val_a.

A small cross-attention reranking head, from the original plan, is built after PetFace arrives and is trained on the PetFace train split only.

## Open-set calibration (2026-10-08)

This replaces the first version, which took the identification threshold from the top scores of unknown val_a queries. That gallery was smaller than the test gallery, so the achieved test FAR overshot the target: 1.96% against 1% for DINOv2 on DogFaceNet development data.

F is the distribution of single impostor scores. It is estimated from every pair of val_a photos that show different dogs.

- **Identification (DIR):** the threshold t solves 1 - F(t)^N = target FAR, with N the test gallery size. That is the chance that an unknown dog's best score passes if its N gallery scores are independent draws from F.
- **Verification (TAR):** the threshold is the single-pair case, F(t) = 1 - target FAR.

Every result reports the achieved test FAR next to the target. A target is flagged as below resolution when it needs a per-impostor rate under one pair in the number of val_a impostor pairs.

## vecstore ablation (2026-10-08)

vecstore had no packaging file, so it could not be installed. A minimal `pyproject.toml` was added on a new branch, `packaging`, in a separate sibling clone, `../vecstore-packaging` (commit 4c4096c, not pushed). The existing working copy, `../vecstore`, was left untouched. The package is installed into this project's venv from that clone and is not listed in `requirements.txt`, since it is not on PyPI.

The ablation runs vecstore brute force and vecstore HNSW at the FAISS HNSW settings (M = 32, efConstruction = 40, the same efSearch sweep), over the same database and queries, timed with the same harness. Recall@10 is measured against exact FAISS search.

Thread pinning (2026-10-08): stage-1 timing runs with NumPy's BLAS pinned to one thread through `VECLIB_MAXIMUM_THREADS=1` (Accelerate; `OPENBLAS_NUM_THREADS=1` on Linux), matching FAISS's single OpenMP thread. The `m2_ann` job refuses to run without the pin, and the first, unpinned vecstore numbers were replaced. Pinned, vecstore brute force still measured faster than FAISS flat at 7,013 vectors (`results/dev/m2_ann.json`). Threading did not cause the gap. The cause is the library underneath: vecstore's brute force is a NumPy matrix-vector product, which this NumPy build runs in Apple Accelerate (`np.show_config()` reports Accelerate as its BLAS), while FAISS uses its own SIMD loop. The win belongs to Accelerate, so it is never used as a resume fact: `pawmatch.report.facts` skips every vecstore field.

## Match service (2026-10-08)

Scope: `POST /match` in this repo only, with no Stay pull request. The gallery is one photo per dog from the DogFaceNet test split (CC BY 4.0), and every response and `GET /about` carry the attribution. Photos are processed in memory and never stored, and no personal data is collected.

Pipeline, fixed before any detection result was measured:

1. OWL-ViT (`google/owlvit-base-patch32`) detects the dog's face zero-shot, with the text "a photo of a dog's face" and a score threshold of 0.1, the transformers default.
2. Alignment is a square crop around the best box, with side 1.2 times its longer edge, at 224 px. It corrects position and scale only, because the detector gives no landmarks.
3. DINOv2 embeds the crop in fp32, and exact search returns the top 5 dogs.
4. Errors: no face gives HTTP 422 with a plain message, a file that is not an image gives 400, and an upload over 10 MB gives 413.
5. The no-match flag uses the Milestone 2 open-set rule: a threshold from the val_a impostor distribution at the gallery size, at a 1% FAR.
6. The optional description is accepted but not used until the Milestone 3 text model exists.

Detector choice: the rule, set beforehand, was to take the faster of OWL-ViT and OWLv2 (`results/dev/detector_speed.json`). Disclosure: in the first timing check, before the rule was applied, one image had no OWL-ViT box above 0.1 where OWLv2 found one. The choice still followed the speed rule. Timing varied widely between runs, but OWL-ViT was faster in every run.

Outcome on DogFaceNet development data (`results/dev/serve_detection.json`): the detector found no face in most clean test queries and gallery photos, and it failed less often on the corrupted queries than on the clean ones. End-to-end rank-1 through the service is far below rank-1 without detection, while rank-1 among detected queries stays close to it. DogFaceNet photos are already tight face crops, which differ from real found-dog photos, so these rates may not carry over. Changing the detector, the threshold, the crop, or adding a fallback when no face is found would rest on these DogFaceNet numbers, and is left to the project owner.

Known mismatch: the no-match threshold is calibrated on val_a embeddings made without detection, while service queries pass through detection and cropping. In a manual check, a correct top match scored below the threshold.

## Cross-attention rerank head (2026-10-08)

Design, fixed before any run (`pawmatch/xattn.py`):

- **Architecture:** frozen DINOv2 patch tokens projected to 128 dimensions, and one 4-head cross-attention layer in which the query tokens attend to the candidate tokens. A residual connection and layer norm follow, then mean pooling and a two-layer MLP to one logit.
- **Training:** on the train split only, with 16 anchors per step. Each anchor is scored against 1 positive (another photo of its dog) and 7 hard negatives (its nearest photos of other dogs by stage-1 embedding), with softmax cross-entropy. AdamW, learning rate 1e-4, weight decay 0.01.
- **Scoring:** the stage-1 top 100 of each test query, fused two ways (head score alone, and RRF with the stage-1 order). The comparisons enter the Benjamini-Hochberg set.

DogFaceNet is capped at 20 steps. After 20 steps the loss stayed near ln 8, the chance level for 8 candidates, and the head scores were uncorrelated with stage-1 similarity, so the DogFaceNet rerank numbers (`results/dev/m2_rerank.json`) only show that the plumbing runs. The PetFace step count is set before the first PetFace run.

## Design rule for pre-cropped inputs (2026-10-09)

The service uses the whole photo as the face crop when no box passes the detection threshold, or when the best box covers more than half of the photo. Benchmark photos such as DogFaceNet are already tight face crops, so the detector often finds no box in them, or a box covering nearly the whole frame. The 60% detection failure on clean DogFaceNet queries measured that property of the test set, not the detector, and nothing is tuned on it. The cutoff of half the photo was fixed in advance.

This replaces the earlier HTTP 422 response for a photo with no detected face. Every response now reports which route produced the crop: `face_box`, `whole_photo_no_box`, or `whole_photo_box_covers_most`. A whole-photo route comes with a plain note.

## Crop and no-match calibration (2026-10-09)

- The face crop expands to a square inside the photo instead of padding with black. Its side is 1.2 times the box's longer edge, capped at the photo's shorter side, and it shifts inward to stay inside the photo.
- One shared path (detect, then the pre-cropped rule, then crop, then embed) produces the service queries, the service gallery (one test photo per dog), and the val_a embeddings that calibrate the no-match threshold. The threshold solves 1 - F(t)^N = 1%, where F is the distribution of val_a impostor scores on that path and N is the gallery size.
- Every `/match` response labels the no-match flag as calibrated on benchmark photos only, not on real found-dog photos.
- The achieved FAR is measured on val_b photos, whose dogs are not in the gallery, and reported next to the target in `results/dev/serve_detection.json`. Nothing is fit on val_b.

## Rerank head on PetFace (2026-10-09)

- The cross-attention head does not count toward the three-run fine-tune budget, since its backbone stays frozen.
- It gets exactly one PetFace run of 5,000 steps, fixed now in `configs/petface.toml` before any PetFace data exists.
- There is no checkpoint selection: the code saves only the final checkpoint, and that checkpoint is the one reported.
- Basis for 5,000 steps: 16 anchors per step makes 80,000 anchors, about two thirds of the estimated 117,844 PetFace training photos (70% of 168,348). Each anchor is scored against 8 candidates, about 640,000 scored pairs in total, for a head with 132,225 parameters.

## Detector chosen on Oxford-IIIT Pet (2026-10-09)

Protocol, fixed before the run:

- The data is the 2,490 Oxford-IIIT Pet dog photos with one head box, split in half by photo with seed 0.
- Both detectors ran at five thresholds (0.01, 0.05, 0.1, 0.2, 0.3), with the pre-cropped rule applied.
- Half A chose by recall at IoU 0.5 of the box the service would use, with ties going to the higher threshold and then to OWL-ViT. Half B reports the choice once.

Choice: OWL-ViT at threshold 0.01 (`results/detection_oxford.json`), which replaces the earlier speed-only choice at 0.1. The half B recall is in the same file.

Two limits:

- **The threshold choice was degenerate.** With dog photos only, a lower threshold can never lose recall, so the lowest grid value won. The threshold's other job, rejecting photos with no dog, cannot be measured without negative photos.
- **Recall is low for both detectors.** With the text "a photo of a dog's face", both often return a box around the whole dog rather than the head, and on half B many photos fall back to the whole photo because the best box covers most of it. A detector trained for dog heads is the likely fix.

## Confidence label instead of a no-match filter (2026-10-13)

- `/match` always returns the top 5 candidates. The old no-match flag is now a confidence label on the top match, `confident_match` or `uncertain`, and nothing is filtered out.
- **Product choice:** the operating point is a 10% gallery-level false-accept rate. The label is `confident_match` when the top score is at or above t, where 1 - F(t)^N = 0.1, F is the distribution of val_a impostor scores through the service path, and N is the gallery size.
- The service computes t when it starts, from the gallery it actually loaded, so the threshold follows the gallery size. Only the val_a impostor scores are cached.
- The benchmark tables still report TAR at FAR 1% and 0.1%; this operating point applies to the service only.
- The achieved gallery-level FAR is measured on val_b photos and on Oxford-IIIT Pet cat photos (`results/dev/serve_detection.json`).

## Trained dog-head detector (2026-10-13)

Fixed before the run (`pawmatch/headdet.py`):

- **Model:** torchvision Faster R-CNN with a MobileNetV3-Large FPN backbone, COCO-pretrained, with a one-class box head. Ultralytics models were excluded because of their AGPL license.
- **Training:** the settings of the torchvision object detection fine-tuning tutorial: SGD, learning rate 0.005, momentum 0.9, weight decay 0.0005, the learning rate divided by 10 every 3 epochs, 10 epochs, batch 2, horizontal flips, and a linear warm-up over the first epoch. Reason: a published recipe for exactly this model family avoids choosing hyperparameters myself. Alternative noted: a longer schedule.
- **Data:** Oxford half A only, split 80/20 by photo with seed 1 into 996 training and 249 internal-validation photos. No checkpoint selection: the final weights are used.
- **Selection, on internal validation only:**
  - The score threshold comes from the same grid as before, chosen by recall at IoU 0.5 under the pre-cropped rule, with ties going to the higher threshold.
  - The service detector, this one or OWL-ViT at 0.01, is chosen by the same recall on the same 249 photos, with ties going to OWL-ViT because it needs no training. OWL-ViT never trained on these photos, so the comparison is fair.
- **Half B:** this detector is reported once, labeled as the second look at half B, next to the 31.00% OWL-ViT baseline.
- **Stop:** after this, there is no more detector work until PetFace results are in.

The model weights were fetched with curl into the torch cache, because this Python build has no certificate bundle; the file's SHA-256 prefix `fb6a3cc7` matches torchvision's published name.

## Cloud path for the PetFace fine-tunes (2026-10-13)

`scripts/cloud_finetune.sh USER@HOST RUN` runs one PetFace fine-tune on a GPU machine that is already running.

1. It syncs the code, configs, frozen split, and git history, with no images in git.
2. It rsyncs the PetFace images, builds the venv, and checks that a CUDA GPU is visible.
3. It runs `make train` for that run: the fine-tune, the run's embeddings, and its Milestone 1 evaluation.
4. It copies the results, checkpoint, embeddings, per-query file, and log back.

The script refuses to run without a frozen PetFace split, with uncommitted changes, or when the run already has a checkpoint, which keeps the three-run budget. It rents, starts, and stops nothing.

To support this, the code now prefers CUDA, then MPS, then CPU. `evaluate.hardware()` reports the platform and GPU on Linux, and each fine-tune checkpoint records its hardware and commit.

Simpler option chosen: images load in the training loop without DataLoader workers, as on the Mac. If GPU utilization turns out low, worker processes are the first speedup to try.

Outcome (2026-10-13, `results/detection_oxford_trained.json`): on half A's internal validation, every threshold from 0.01 to 0.2 tied at the top, so the tie rule chose 0.2. The trained detector beat OWL-ViT there by a wide margin, so the service now uses it (`detect.DETECTOR = "frcnn_head"`, threshold 0.2). Half B, the second look, confirms the gain over the 31.00% OWL-ViT baseline. Caveat: the detector trains and tests on Oxford photos, so this is in-distribution recall; transfer to real found-dog photos is unmeasured. Detector work now stops until PetFace results are in.

Outcome of the 10% operating point (2026-10-13, `results/dev/serve_detection.json`, development data): the achieved gallery-level FAR is well under the target on val_b, and no Oxford cat photo was labeled a confident match. The trained detector still finds a "head" box on most cat photos, because it learned only dogs, so the confidence label, not the detector, is what keeps cats out. Under corruptions far fewer correct matches are labeled confident than on clean photos.

## Enforcing the PetFace terms (2026-10-14)

- **Remote cleanup:** the cloud script deletes `data/PetFace` on every exit, including failures, through a shell exit hook. It also deletes the PetFace-derived caches on the remote, but only after the results and checkpoint are confirmed on this machine, so a failed copy can be retried. The terms asked only for the images to be removed; also removing derived caches is the stricter of the two defensible choices, and it costs nothing once the copies are safe.
- **No-redistribution test:** it allows images only under `docs/images/dogfacenet/`, a folder for DogFaceNet demo images. The alternative, banning all images, was rejected because it would block DogFaceNet demo images, which the terms allow. The test skips when run outside a git checkout.
- **Checkpoint terms:** the non-commercial marking lives in each checkpoint's metadata as a `terms` field, added by `pawmatch.data.model_terms`. A separate license file per checkpoint was not needed, since checkpoints never leave `data/cache/`.

## Serving Stay (2026-10-14)

- **How Stay connects:** the Stay page calls the PawMatch service directly from the browser. No photo passes through or is stored by Stay's backend. The service allows Stay's origin through CORS (`PAWMATCH_ALLOWED_ORIGINS`).
- **Gallery photos:** only a DogFaceNet gallery's photos are served (`GET /gallery/{dog_id}.jpg`, CC BY 4.0, attributed in every response). With a PetFace gallery the route returns 404 and matches carry IDs and scores only, even if photo paths are supplied. Every response names its gallery, so a client can apply the same rule.
- **Switching models:** `[serve] checkpoint` is the one config line that swaps in a fine-tuned model, such as a PetFace run, while the gallery stays DogFaceNet. Until it is set, the service uses no PetFace data.
- **No description field:** the optional description is not collected on the Stay page, since free text can carry personal data and the text model does not exist yet.

## Int8 quantization of the service models (2026-10-27)

A requested measurement of int8 copies of the two service models against fp32. The service keeps its fp32 models, and no detector is chosen, so the detector freeze above still holds. Fixed before any held-out measurement (`pawmatch/quantize.py`):

- **Engine:** PyTorch post-training quantization on the CPU with qnnpack, the only int8 engine on arm64 (fbgemm is x86 only). Quantized kernels have no MPS version, so fp32 is measured on the CPU too. ONNX Runtime is not installed, and PyTorch ran cleanly, so no dependency was added.
- **Detector:** static int8 for the MobileNetV3 body and the FPN (FX graph mode, PyTorch's default qnnpack settings), calibrated on the first 100 half-A training photos, and dynamic int8 for the box head's Linear layers. Frozen batch norm becomes plain batch norm so that it folds into the convolutions. QNNPACK rejects the FPN's extra 1x1 stride-2 max pool, so that one op stays fp32, as do the RPN head, RoIAlign, and NMS. The score threshold stays 0.2. Reason for static: the backbone takes most of the detector's CPU time, and dynamic quantization reaches only Linear layers.
- **Re-ID:** the model the service uses, frozen DINOv2 ViT-S/14 (`[serve] model`, no checkpoint), with dynamic int8 Linear layers at the default settings, the standard route for transformers. LayerNorm, softmax, and GELU stay fp32.
- **Measures:** serialized state_dict size; p50 and p95 latency at batch 1, forward pass only, over 100 photos drawn with seed 0, as the median of 5 runs with fp32 and int8 interleaved; detector recall at IoU 0.5 under the pre-cropped rule on half A internal validation and on half B; re-ID with the Milestone 1 protocol on the DogFaceNet test split.
- **Looks:** half B gets its third look, labeled as such. The DogFaceNet test split is development data that has been looked at before. Nothing is chosen from either.
- **Where results go:** the detector numbers (Oxford) in `results/quantization.json`; the re-ID numbers (DogFaceNet, development data) in `results/dev/quantization_reid.json`, which the README generator never reads; both in `results/quantization.md`. The code stamp checks `pawmatch/` only, since these runs use no C++ code.
- **Disclosure:** before the protocol was fixed, a probe timed both models on the CPU and ran the int8 detector on three half-A internal-validation photos to check that it runs. It found the max-pool refusal, and it showed dynamic int8 DINOv2 running slower than fp32. Nothing was tuned on it.
