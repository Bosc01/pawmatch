CONFIG ?= configs/dogfacenet.toml
RUN ?= arcface_bh
PORT ?= 8000
PY = venv/bin/python
BACKBONES = resnet50_in1k clip_vitb16 dinov2_vits14

.PHONY: data embed eval train rerank-head retrieve precision-test detect serve export cpp cpp-test bench test readme facts

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

rerank-head:
	$(PY) -m pawmatch.xattn $(CONFIG) train
	$(PY) -m pawmatch.xattn $(CONFIG) score

retrieve:
	VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PY) -m pawmatch.retrieve $(CONFIG) m2_ann
	$(PY) -m pawmatch.retrieve $(CONFIG) m2_rerank m2_open_set m2_corruption
	$(PY) -m pawmatch.significance $(CONFIG)

detect:
	$(PY) -m pawmatch.detect $(CONFIG) speed
	$(PY) -m pawmatch.detect $(CONFIG)

serve:
	PAWMATCH_CONFIG=$(CONFIG) venv/bin/uvicorn --factory pawmatch.serve:build_app --port $(PORT)

precision-test:
	$(PY) -m pawmatch.train $(CONFIG) precision_test

export:
	$(PY) -m pawmatch.export $(CONFIG)

cpp:
	cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
	cmake --build cpp/build -j

cpp-test: cpp
	ctest --test-dir cpp/build --output-on-failure

bench: cpp
	cpp/build/bench data/cache/m4/gallery.bin data/cache/m4/queries.bin > data/cache/m4/bench.json
	$(PY) -m pawmatch.export --report data/cache/m4/bench.json

test:
	$(PY) -m pytest -q

readme:
	$(PY) -m pawmatch.report readme

facts:
	$(PY) -m pawmatch.report facts
