#!/usr/bin/env bash
# One PetFace fine-tune on a rented Linux GPU machine, with the results copied back.
#
# Usage, from the repo root on this Mac:
#   scripts/cloud_finetune.sh USER@HOST RUN
# RUN is arcface, arcface_bh, or augmented (run (c): set train.run_c_arm and commit first).
# The machine must already be running. This script rents, starts, and stops nothing.
#
# PetFace terms (DATA.md, 2026-10-14): cloud copies are allowed for training only. The
# machine must stay private (no shared users, public images, or snapshots). This script
# deletes data/PetFace from the machine when it exits, whether or not the run succeeded,
# and deletes the PetFace-derived caches there once the results are safely copied back.
#
# Needs:
#   - an Ubuntu machine with one NVIDIA GPU of 24 GB or more and its driver installed,
#     with python3 (venv), rsync, make, and git, reachable over SSH with a key;
#   - about 60 GB of free disk for the dataset, the venv, and caches;
#   - PetFace in data/PetFace on this Mac, and its split frozen and committed
#     (make data CONFIG=configs/petface.toml), so the remote run checks the same hash;
#   - a clean git tree, so the results carry a real commit.
# Copies back: results/m1_RUN.json, and into data/cache/petface the checkpoint, the
# run's embeddings, its per-query file, and the log.
set -euo pipefail

host=${1:?usage: scripts/cloud_finetune.sh USER@HOST RUN}
run=${2:?usage: scripts/cloud_finetune.sh USER@HOST RUN}
case "$run" in arcface | arcface_bh | augmented) ;; *) echo "RUN must be arcface, arcface_bh, or augmented" >&2; exit 2 ;; esac
[ -f splits/petface.json ] || { echo "freeze the PetFace split first: make data CONFIG=configs/petface.toml" >&2; exit 2; }
[ -d data/PetFace ] || { echo "data/PetFace is missing on this machine" >&2; exit 2; }
[ -z "$(git status --porcelain)" ] || { echo "commit local changes first, so the results carry a real commit" >&2; exit 2; }
if [ -e "data/cache/petface/ckpt/$run.pt" ]; then
    echo "run $run already has a checkpoint; each PetFace run is used once (decisions.md)" >&2
    exit 2
fi

dir=pawmatch
remove_petface() {
    if ssh "$host" "rm -rf $dir/data/PetFace"; then
        echo "== deleted data/PetFace from $host"
    else
        echo "WARNING: could not delete data/PetFace from $host; delete it by hand (PetFace terms)" >&2
    fi
}
echo "== code, configs, frozen split, and git history (no images, caches, or results)"
rsync -az --delete --exclude data/ --exclude venv/ --exclude 'cpp/build*' --exclude results/ ./ "$host:$dir/"
echo "== PetFace images"
trap remove_petface EXIT
rsync -az data/PetFace/ "$host:$dir/data/PetFace/"
echo "== environment"
ssh "$host" "cd $dir && python3 -m venv venv && venv/bin/pip install -q --upgrade pip \
    && venv/bin/pip install -q -r requirements.txt \
    && venv/bin/python -c 'import torch; assert torch.cuda.is_available(), \"no CUDA GPU visible\"; print(torch.cuda.get_device_name())'"
echo "== fine-tune $run, then its embeddings and Milestone 1 evaluation"
ssh "$host" "cd $dir && set -o pipefail && make train CONFIG=configs/petface.toml RUN=$run 2>&1 | tee train_$run.log"
echo "== copy results back"
mkdir -p results data/cache/petface/ckpt data/cache/petface/per_query data/cache/petface/logs
rsync -az "$host:$dir/results/m1_$run.json" results/
rsync -az "$host:$dir/data/cache/petface/ckpt/$run.pt" data/cache/petface/ckpt/
rsync -az "$host:$dir/data/cache/petface/$run/" "data/cache/petface/$run/"
rsync -az "$host:$dir/data/cache/petface/per_query/m1_$run.npz" data/cache/petface/per_query/
rsync -az "$host:$dir/train_$run.log" data/cache/petface/logs/
if [ -s "results/m1_$run.json" ] && [ -s "data/cache/petface/ckpt/$run.pt" ]; then
    ssh "$host" "rm -rf $dir/data/cache/petface" && echo "== deleted PetFace-derived caches from $host"
else
    echo "results did not all copy back; remote caches kept so the copy can be retried" >&2
    exit 1
fi
echo "== done: results/m1_$run.json. The machine is still running; stop it in the provider console."
