# PawMatch

Open-set dog re-identification for lost-pet reunification. Work in progress: the pipeline runs, and no PetFace results exist yet.

## Results

<!-- results:start -->
No PetFace results yet.
<!-- results:end -->

`make readme` generates the tables above from `results/*.json`. DogFaceNet runs used to build and test the pipeline are written to `results/dev/` and never appear here; `tests/test_report.py` enforces that.

## Running

```
make data      # build, check, and freeze the identity-disjoint split
make embed     # cache frozen-backbone embeddings
make eval      # Milestone 1 metrics with bootstrap intervals
make train     # fine-tune (smoke steps only outside PetFace)
make test
```

Every target reads `CONFIG`, which defaults to `configs/dogfacenet.toml`. Use `CONFIG=configs/petface.toml` for PetFace.

## Data

See [DATA.md](DATA.md).
