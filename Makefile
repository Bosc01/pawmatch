CONFIG ?= configs/dogfacenet.toml
RUN ?= arcface_bh
PY = venv/bin/python
BACKBONES = resnet50_in1k clip_vitb16 dinov2_vits14

.PHONY: data embed eval train retrieve precision-test test readme facts

data:
	$(PY) -m pawmatch.data $(CONFIG)

embed:
	for b in $(BACKBONES); do $(PY) -m pawmatch.embed $(CONFIG) $$b --corrupt || exit 1; done
	$(PY) -m pawmatch.embed $(CONFIG) dinov2_vits14 --splits train --patches
	$(PY) -m pawmatch.embed $(CONFIG) resnet50_in1k --splits test --patches

eval:
	for b in $(BACKBONES); do $(PY) -m pawmatch.evaluate $(CONFIG) $$b || exit 1; done

train:
	$(PY) -m pawmatch.train $(CONFIG) $(RUN)
	$(PY) -m pawmatch.embed $(CONFIG) dinov2_vits14 --run $(RUN) --corrupt
	$(PY) -m pawmatch.evaluate $(CONFIG) $(RUN)

retrieve:
	$(PY) -m pawmatch.retrieve $(CONFIG)
	$(PY) -m pawmatch.significance $(CONFIG)

precision-test:
	$(PY) -m pawmatch.train $(CONFIG) precision_test

test:
	$(PY) -m pytest -q

readme:
	$(PY) -m pawmatch.report readme

facts:
	$(PY) -m pawmatch.report facts
