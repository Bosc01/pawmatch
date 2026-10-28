# CUDA benchmark on Google Colab

How the CUDA scorer (`cpp/bench_cuda.cu`) is built, checked against the CPU scorer, and benchmarked on a Colab T4 runtime. `scripts/run_colab.sh` does the work; the cells below only move files.

## On the Mac: three files to upload

Run from a terminal. This zips the working tree as it is, uncommitted changes included, leaving out everything git ignores (`data/`, `venv/`, `cpp/build*/`, `docs/private/`). It puts the zip and the two fixture files in one folder, so a single upload dialog can take all three.

```bash
cd ~/pawmatch && rm -rf ~/Desktop/pawmatch-colab && mkdir ~/Desktop/pawmatch-colab && git ls-files -co --exclude-standard | zip -q ~/Desktop/pawmatch-colab/pawmatch.zip -@ && cp data/cache/m4/gallery.bin data/cache/m4/queries.bin ~/Desktop/pawmatch-colab/
```

The fixture files hold DogFaceNet embeddings (CC BY 4.0) and synthetic vectors, no PetFace data.

## In Colab

Open a new notebook, then go to Runtime > Change runtime type > T4 GPU, and save. Paste each cell below into its own code cell and run them in order.

Cell 1, confirm the GPU:

```
!nvidia-smi
```

Cell 2, upload. Select `pawmatch.zip`, `gallery.bin`, and `queries.bin` from `~/Desktop/pawmatch-colab` in one dialog. The result is assigned to a variable so the notebook does not try to print 92 MB of file contents.

```python
from google.colab import files
uploaded = files.upload()
print(sorted(uploaded))
```

Cell 3, unzip the repo and put the fixtures where the benchmark reads them:

```
%%bash
set -e
rm -rf /content/pawmatch
unzip -q /content/pawmatch.zip -d /content/pawmatch
mkdir -p /content/pawmatch/data/cache/m4
mv /content/gallery.bin /content/queries.bin /content/pawmatch/data/cache/m4/
ls -l /content/pawmatch/data/cache/m4
```

If Colab saved an upload as `gallery (1).bin`, a file was uploaded twice. Use Runtime > Disconnect and delete runtime, then start again from cell 1.

Cell 4, build and run. This installs CMake 3.24 or later if the runtime has an older one, builds the CUDA target, runs the exactness check and then the benchmark (`make bench-cuda`), records `nvidia-smi` and `lscpu`, and writes `/content/pawmatch/results_colab.zip`. The CPU scalar pass at 100K rows is the slow part. Keep the tab open until the cell finishes.

```
!bash /content/pawmatch/scripts/run_colab.sh
```

Cell 5, download the results:

```python
from google.colab import files
files.download('/content/pawmatch/results_colab.zip')
```

## Back on the Mac

The zip holds:

- `bench_cuda.json`: the exactness result and every run's p50 and p99, with the median of 5 runs;
- `run.log`: tool versions, the build output, and the benchmark's progress lines;
- `nvidia-smi.txt` and `lscpu.txt`: the GPU and CPU models;
- `sha256.txt`: hashes of the fixtures and the scorer source, to tie the numbers to the code;
- `NOTE.txt`: the vCPU count, and the note that the CPU timings come from a shared VM.

If the build or the exactness check fails, the script still writes the zip, and `run.log` shows the error.
