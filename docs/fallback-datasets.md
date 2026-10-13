# Fallback datasets for dog re-identification

Openly licensed dog datasets with individual identities that could stand in for PetFace if access never comes. Searched on 2026-10-13 with web search only; nothing was downloaded. Licenses are as stated on each dataset's own page and should be checked again before use.

## Summary

| Dataset | License as stated | Dogs | Images | Photos | Access | Published baseline |
|---|---|---|---|---|---|---|
| YT-BB-Dog | CC BY 4.0 (project page); frames come from YouTube videos | 2,723 | 27,036 | Whole-dog box crops from video frames | Direct zip, 600 MB | Not retrieved (see notes) |
| DogFaceNet, 224 resized | CC BY 4.0 (Zenodo) | 1,393 | 8,363 | Aligned face crops | Direct download | 92% and 86% pair verification accuracy (see notes) |
| MPDD (Multi-Pose Dog Dataset) | CC BY 4.0 (Mendeley Data) | 192 | 1,657 | Phone photos in varied poses | Direct download | Not retrieved |
| Sibetan | CC BY 4.0 (project page) | 59 | 1,755 | Whole-dog box crops from camera-trap video | Direct zip, 12 MB | Rank-1 gain of more than 9 points over the paper's baseline; absolute numbers not retrieved |
| WildlifeReID-10k (dog part) | No single license; components "have permissible licenses for being re-uploaded" | 1,393 per the paper | 8,363 per the paper | Same as DogFaceNet | Kaggle | No dog-only numbers |
| Flickr-dog | Flickr Creative Commons photos | 42 | 374 | Aligned face crops | No public release found | 67.6% balanced accuracy, 10-fold cross-validation |
| Snoopybook | Not stated | 18 | 251 | Registered face images | No public release found | 89.4% balanced accuracy, 10-fold cross-validation |
| Pet Biometric Challenge 2022 | Not found | About 6,000 (training) | About 20,000 (training) | Nose-print close-ups | Challenge release | 86.67% AUC on test pairs (third place) |

For scale: PetFace has 46,755 dogs and 168,348 images (DATA.md). No openly licensed set found here comes within an order of magnitude of that.

## Notes per dataset

**YT-BB-Dog and Sibetan.** Both come from the same 2025 Ecological Informatics paper, "Background-invariant re-identification of dogs from camera-trap videos in non-controlled environments", and are hosted at https://www.lirmm.fr/YT-BB-Dog_Sibetan/. The page states that all its datasets are licensed under CC BY 4.0. YT-BB-Dog is built from YouTube-BoundingBoxes (Real et al., CVPR 2017), so its images are frames of other people's YouTube videos. Whether the page's license covers those frames is unclear, and that question would have to be settled before relying on the set. The abstract reports that the paper's method improves rank-1 on Sibetan by more than 9 points over its baseline. The paper's results tables sit behind a bot-protection page, so absolute numbers were not retrieved. The images are crops of whole dogs, not faces, which changes the task from face matching to body matching.

**DogFaceNet.** Source: https://zenodo.org/records/12578449 (CC BY 4.0), from Mougeot, Li, and Jia, "A Deep Learning Approach for Dog Face Verification and Recognition", PRICAI 2019. The project README reports 92% accuracy on pair verification over an open set of 48 unknown dogs. That was measured on an earlier version of about 1,400 images; the current, larger version gives 86%. The raw "large" variant has 2,522 folders but is not fully curated, and some folders hold more than one dog. This project already uses DogFaceNet for pipeline development, and the decisions so far reserve it for plumbing and the final cross-dataset evaluation. Making it the main benchmark would reverse that decision.

**MPDD.** Source: https://data.mendeley.com/datasets/v5j6m8dzhv/1 (CC BY 4.0, version 1, 19 June 2023, DOI 10.17632/v5j6m8dzhv.1). The photos were taken with one phone to vary pose and viewpoint, and the paper groups them as standing, sitting, and lying. The record does not say whether the images are face crops, but the pose focus suggests whole-dog photos. Paper: He et al., ICASSP 2023 (MPFNet). Its numbers were not retrieved.

**WildlifeReID-10k.** Adam et al., CVPR 2025 Workshops, hosted on Kaggle. The paper lists DogFaceNet as its only dog source (1,393 dogs, 8,363 images), so it adds no new dogs. One secondary source gives 1,584 dogs and 10,020 images; those totals equal DogFaceNet plus MPDD, so a later release may include MPDD as well. That is unverified. The paper reports only metrics averaged across species.

**Flickr-dog and Snoopybook.** Moreira et al., "Where Is My Puppy? Retrieving Lost Dogs by Facial Features", Multimedia Tools and Applications, 2017 (arXiv 1510.02781). Flickr-dog has 21 pugs and 21 huskies. The paper said the data would be released on acceptance, but no public copy was found. The baselines are the paper's best balanced accuracies under 10-fold stratified cross-validation.

**Pet Biometric Challenge 2022.** This was a CVPR 2022 Biometrics Workshop challenge, with the data collected by Ant Group. The counts and the 86.67% AUC come from participants' papers (arXiv 2206.00580 and 2205.15934). No license statement was found. The images are nose prints, a different identification modality from face or body photos.

## Assessment

If PetFace never arrives, no single open set replaces it.

The strongest candidate is YT-BB-Dog. It has the most dogs and images and a stated CC BY 4.0 license, but its frames come from YouTube videos, so there is an open license question. Under the stopping rules, that question goes to the project owner before the set is used.

MPDD and Sibetan are small, openly licensed test sets that could serve as extra evaluations. DogFaceNet stays the cross-dataset evaluation unless the earlier decision is reversed.

Moving to YT-BB-Dog would also change the task from faces to whole-dog crops. The service's face-crop step would then need to become a dog-body crop.
