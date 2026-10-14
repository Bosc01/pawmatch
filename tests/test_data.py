import pytest

from pawmatch import data


@pytest.fixture
def dogfacenet(tmp_path, monkeypatch):
    """100 fake dogs with 2, 3, or 4 photos each (299 photos)."""
    for d in range(100):
        (tmp_path / "imgs" / str(d)).mkdir(parents=True)
        for k in range(2 + d % 3):
            (tmp_path / "imgs" / str(d) / f"{d}.{k}.jpg").touch()
    monkeypatch.setattr(data, "FROZEN_DIR", tmp_path / "splits")
    return {"dataset": {"name": "dogfacenet", "images": str(tmp_path / "imgs"), "seed": 0}}


def test_dogfacenet_split_is_identity_disjoint_frozen_and_keeps_two_photo_dogs(dogfacenet):
    splits, sha = data.load_splits(dogfacenet, freeze=True)
    dogs = {s: {d for _, d in splits[s]} for s in data.SPLITS}
    assert [len(dogs[s]) for s in data.SPLITS] == [70, 5, 5, 20]
    assert sum(len(v) for v in splits.values()) == 299
    assert data.load_splits(dogfacenet)[1] == sha
    dogfacenet["dataset"]["seed"] = 1
    with pytest.raises(SystemExit, match="no longer match"):
        data.load_splits(dogfacenet)


def test_leakage_guard_rejects_a_dog_in_two_splits():
    splits = {s: [] for s in data.SPLITS}
    splits["train"], splits["test"] = [("x/7/a.jpg", "7")], [("x/7/b.jpg", "7")]
    with pytest.raises(ValueError, match="dog 7 is in both train and test"):
        data.check_disjoint(splits)


def test_petface_reads_official_split_files(tmp_path, monkeypatch):
    split = tmp_path / "split"
    split.mkdir()
    (split / "train.csv").write_text("filename,label\ndog/000001/00.png,0\ndog/000001/01.png,0\ndog/000002/00.png,1\n")
    (split / "val.txt").write_text("\n".join(f"dog/0000{d}/00.png" for d in (10, 11, 12, 13)) + "\n")
    (split / "test.txt").write_text("dog/000020/00.png\ndog/000020/01.png\n")
    (split / "verification.csv").write_text("filename1,filename2,label\ndog/000020/00.png,dog/000020/01.png,1\n")
    monkeypatch.setattr(data, "FROZEN_DIR", tmp_path / "splits")
    cfg = {"dataset": {"name": "petface", "images": "imgs", "split_dir": str(split), "seed": 0,
                       "verification_csv": str(split / "verification.csv")}}
    splits, _ = data.load_splits(cfg, freeze=True)
    assert {d for _, d in splits["train"]} == {"000001", "000002"}
    assert splits["test"] == [("imgs/dog/000020/00.png", "000020"), ("imgs/dog/000020/01.png", "000020")]
    val_a, val_b = ({d for _, d in splits[s]} for s in ("val_a", "val_b"))
    assert len(val_a) == len(val_b) == 2 and val_a | val_b == {"000010", "000011", "000012", "000013"}
    assert data.official_pairs(cfg, splits["test"]) == [(0, 1, 1)]


ITEMS = [(f"{d}/{k}.jpg", str(d)) for d in range(10) for k in range(2 + d % 2)]


def test_gallery_query_closed_and_open_set():
    dogs_of = lambda ix: [ITEMS[i][1] for i in ix]
    g, q, unknown = data.gallery_query(ITEMS, seed=0)
    assert sorted(dogs_of(g)) == sorted({d for _, d in ITEMS}) and not unknown
    assert set(g).isdisjoint(q) and len(g) + len(q) == len(ITEMS)
    g_open, q_open, unknown = data.gallery_query(ITEMS, seed=0, unknown_frac=0.5)
    assert q_open == q and len(unknown) == 5
    assert set(dogs_of(g_open)) == {d for _, d in ITEMS} - unknown
    assert set(g_open) <= set(g)  # known dogs keep the same gallery photo


def test_verification_pairs_one_positive_and_one_negative_per_dog():
    pairs = data.verification_pairs(ITEMS, seed=0)
    assert sorted(y for _, _, y in pairs) == [0] * 10 + [1] * 10
    for i, j, y in pairs:
        assert i != j and (ITEMS[i][1] == ITEMS[j][1]) == bool(y)


def test_petface_trained_models_are_marked_non_commercial():
    assert "non-commercial" in data.model_terms({"dataset": {"name": "petface"}})
    assert data.model_terms({"dataset": {"name": "dogfacenet"}}) is None
