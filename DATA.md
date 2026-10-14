# Data

No images are stored in this repository. Everything under `data/` is ignored by git.

## PetFace (primary dataset)

- Paper: Shinoda and Shiohara, "PetFace: A Large-Scale Dataset and Benchmark for Animal Identification", ECCV 2024, arXiv 2407.13555.
- Project page: https://dahlian00.github.io/PetFacePage/
- Code: https://github.com/mapooon/PetFace

### Access

Status: the authors replied with usage terms on 2026-10-14 (below). The data is not yet in `data/PetFace`.

Access is by Google Form only: https://docs.google.com/forms/d/e/1FAIpQLSfRPJaCmU6oQ4X_uB6H-EM5MSeczKczZxbQ5H9FMRS4KNY59w/viewform

The form asks for an email, full name, affiliation, the purpose of use, and optional comments. The authors review each form by hand and email a download link. Their README says they do not reply when:

- any information is missing,
- the email is not an institutional address (Gmail is named as an example),
- the form is not written in English, or
- the requesting company does not publish research papers.

If no link arrives within a week, the README says to submit the form again and not to open a GitHub issue.

### License and terms

There is no formal license. The GitHub repository has no license file, and the paper and project page name none. The terms that do exist:

- The repository README states that the dataset, the code, and the pretrained models are "for non-commercial research purpose only".
- The project page states that the dataset is for research purposes only, and that the form exists to confirm research use.
- The project page is licensed CC BY-SA 4.0. That license covers the website, not the dataset.
- Redistribution is not permitted, per the authors' reply below.

### Terms from the authors' reply (received 2026-10-14)

As relayed by the project owner on October 14, 2026. The authors' original wording is not reproduced in this repository.

> 1. "Cloud copies are allowed for training."
> 2. "Non-commercial use only, including models trained on PetFace."
> 3. "No redistribution."
> 4. "Display is view-only."

How the project enforces them:

1. **Cloud copies:** `scripts/cloud_finetune.sh` deletes `data/PetFace` from the rented machine whenever the script exits, and deletes the PetFace-derived caches there once the results are copied back. The rented machine stays private: no shared users, public images, or snapshots.
2. **Non-commercial:** every PetFace-trained checkpoint (the fine-tune arms and the rerank head) carries a `terms` field marking it non-commercial (`pawmatch.data.model_terms`). The README says so too.
3. **No redistribution:** `tests/test_no_petface_data.py` fails if git tracks anything under `data/`, any embedding cache, patch-feature file, or checkpoint, any PetFace-named data file, or any image outside `docs/images/dogfacenet/`. It also fails if a README or doc embeds such an image. PetFace sample images never appear in the README or docs.
4. **View-only display:** with a PetFace gallery, `/match` returns entries as IDs and scores only, never image bytes or image URLs, and a test checks the fields. Demo images shown to users come from DogFaceNet only.

Milestone 5 (Stay) stays on hold until the project owner confirms whether Stay has any revenue, ads, or company behind it.

### Dog subset (figures from the paper, Table 4b)

- 46,755 individuals, 168,348 images.
- Images are aligned face crops at 224 x 224.
- Annotations for dogs: sex and breed, collected from the source websites, so coverage per dog is not known until the files arrive. The paper (Appendix A) lists color and pattern annotations for eleven animal classes, and Dog is not one of them.

### Published dog baselines and their protocols

The paper defines two protocols, and they are not interchangeable.

| Protocol | Identities at test time | Dog result (ArcFace) |
|---|---|---|
| Re-identification, top-1 (Table 2) | Seen in training, scored as classification over all 46,755 dogs | 77.86% |
| Verification AUC (Table 3) | Unseen, one positive and one negative pair per test identity | 99.45% |

Other dog verification AUCs in Table 3: ImageNet 97.18%, CLIP 91.86%, MegaDescriptor 93.75%.

The paper splits each family 7:1:2 into train, validation, and test, with sources kept disjoint. The official split files ship with the dataset under `split/dog/`.

### Expected local layout

```
data/PetFace/
  images/dog/<id>/<nn>.png
  split/dog/{train.csv, val.txt, test.txt, reidentification.csv, verification.csv}
  annotations/dog.csv
```

## DogFaceNet (secondary dataset)

### License and attribution

Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. No changes are made to the images; any derived files (embeddings, crops) are marked as derived from this dataset.

Dataset: Guillaume Mougeot, "The DogFaceNet datasets", version 1, Zenodo, 2024. DOI 10.5281/zenodo.12578449. https://zenodo.org/records/12578449

Paper: Guillaume Mougeot, Dewei Li, and Shuai Jia, "A Deep Learning Approach for Dog Face Verification and Recognition", PRICAI 2019: Trends in Artificial Intelligence, Springer, pp. 418 to 430. DOI 10.1007/978-3-030-29894-4_34.

Code: https://github.com/GuillaumeMougeot/DogFaceNet (MIT).

### Use in this project

Pipeline building, tests, and a later cross-dataset evaluation only. No model, hyperparameter, or threshold is tuned on it.

### Local copy (downloaded 2026-10-08)

- File: `DogFaceNet_224resized.zip`, 75,571,850 bytes, MD5 `010c207e202bb499039452aa7363b015` (matches Zenodo).
- Contents, counted locally: 1,393 dogs, 8,363 JPEG images, 2 to 41 images per dog. No other file types.
- No breed, sex, or color annotations.

```
data/dogfacenet/after_4_bis/<id>/<id>.<n>.jpg
```

## Decisions (2026-10-08)

### Splits and protocols (PetFace dogs)

- Use the official split files in `split/dog/`. No new split is drawn.
- The official validation dogs are split in half by identity. Val-A calibrates the no-match threshold (Milestone 2). Val-B trains the learned ranker (Milestone 3). The split is frozen and hashed.
- Comparison with the paper: verification AUC on the official `verification.csv`, against Table 3.
- Gallery/query and open-set results run on the official test dogs and are reported as protocols defined by this project. They are not compared with the 77.86% re-identification figure, which measures a different task.

### Text attributes (Milestone 3)

- Main results use breed and sex only, the two attributes PetFace provides for dogs. Descriptions generated from them are SYNTHETIC.
- Coat color is an ablation only. Each dog gets one color label, derived from its photos that are not in the gallery, by a model that is not one of the retrieval backbones (ImageNet ResNet-50, CLIP ViT-B/16, DINOv2 ViT-S/14). These labels are marked MODEL-DERIVED wherever they appear.
- 200 dogs are hand-labeled by the author to measure the accuracy of the color labels.
- The ranker is reported with and without color.

### Stay integration (Milestone 5)

On hold until the PetFace authors confirm whether their research-only terms allow it.

## Oxford-IIIT Pet (detector evaluation only, downloaded 2026-10-09)

- Source: https://www.robots.ox.ac.uk/~vgg/data/pets/ (`images.tar.gz`, `annotations.tar.gz`).
- License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0), https://creativecommons.org/licenses/by-sa/4.0/. The project page states that the copyright of the images remains with their original owners. No images are committed or redistributed; the dataset is used only to measure dog face detection.
- Citation: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
- Local copy: `images.tar.gz` SHA-256 `67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d`, `annotations.tar.gz` SHA-256 `52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91`, extracted under `data/oxford_pets/`.
- Use: dog images only (species 2 in `annotations/list.txt`) that have a head bounding box in `annotations/xmls/`. They are split in half by image with a fixed seed. Half A chooses the detector and threshold; half B reports recall at IoU 0.5.
- Trained model: a dog-head detector (`pawmatch/headdet.py`) is trained on Oxford half A, so its weights derive from CC BY-SA 4.0 data. It starts from torchvision's COCO-pretrained Faster R-CNN weights, and torchvision notes that pretrained weights may carry terms from their training data. The weights stay in `data/cache/` and are not committed or distributed. Whether ShareAlike applies to trained weights is not settled, and it must be resolved before the detector ships anywhere, for example in Stay.
