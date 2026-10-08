# Data

No images are stored in this repository. Everything under `data/` is ignored by git.

## PetFace (primary dataset)

- Paper: Shinoda and Shiohara, "PetFace: A Large-Scale Dataset and Benchmark for Animal Identification", ECCV 2024, arXiv 2407.13555.
- Project page: https://dahlian00.github.io/PetFacePage/
- Code: https://github.com/mapooon/PetFace

### Access

Status: no download link yet (2026-10-08).

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
- Redistribution is not addressed. This project treats it as not permitted: no PetFace images are committed, published, or served outside this machine.

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
