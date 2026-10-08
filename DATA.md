# Data

No images are stored in this repository. Everything under `data/` is ignored by git.

## PetFace (primary dataset)

- Paper: Shinoda and Shiohara, "PetFace: A Large-Scale Dataset and Benchmark for Animal Identification", ECCV 2024, arXiv 2407.13555.
- Project page: https://dahlian00.github.io/PetFacePage/
- Code: https://github.com/mapooon/PetFace

### Access

Status: not yet requested (2026-10-08).

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

## DogFaceNet (fallback, not downloaded)

- Source: https://zenodo.org/records/12578449 (code at https://github.com/GuillaumeMougeot/DogFaceNet)
- License: CC BY 4.0. Open download, no form.
- Size: 1,393 dogs in the 224 x 224 resized set (75.6 MB zip); 2,522 folders in the large set (1.8 GB zip). The Zenodo page warns that some large-set folders contain more than one dog.
- No breed or color annotations.
