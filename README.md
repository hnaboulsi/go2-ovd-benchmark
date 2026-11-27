# Go2 Open-Vocabulary Detection Benchmark

I built this ICON Lab experiment to compare object detectors on the same frames from a walking
Unitree Go2 in MuJoCo. The simulator records RGB, segmentation masks, and camera poses. Grounding
DINO runs offline, and the evaluator scores detections against segmentation-derived ground truth.

The retained reproduction run completed 12 episodes and processed 240 frames. It reproduced the
reference run's frames, ground truth, detections, and evaluation metrics other than latency.
The experiment used two YCB objects, a baseball and a softball. Use the commands below to generate
a fresh run.

The simulation uses [Elijah Chan's convex MPC controller](https://github.com/elijah-waichong-chan/go2-convex-mpc)
with the pinned revision and patch described in [controller/upstream.json](controller/upstream.json).
The project and upstream notices are in [LICENSE](LICENSE) and [LICENSES](LICENSES).

## Architecture

```text
scripted velocity -> convex MPC -> MuJoCo actuators
                              |
                              v
                      front_rgb at 5 Hz
                              |
                              v
       lossless RGB + segmentation + poses + manifests
                              |
                              v
        offline detector adapter -> immutable JSONL
                              |
                              v
              evaluator -> report and overlays
                              |
                              v
          future rejection module (not implemented)
```

## Bootstrap

On an Apple Silicon Mac, run:

```bash
./scripts/bootstrap_macos.sh
```

The script verifies its host, downloads a pinned micromamba binary, clones and verifies the controller commit, verifies and applies the patch, creates an isolated Python 3.10 simulation environment, creates an isolated Python 3.11 perception environment from the hash-locked requirements, downloads only the required YCB files, resolves the pinned detector snapshot, and runs controller, camera, and detector smoke checks. Downloaded dependencies, weights, environments, generated runs, and reports are ignored by Git.

For simulator commands:

```bash
./.tools/micromamba run -p "$PWD/.envs/simulation" go2-ovd validate-controller --repeats 3
./.tools/micromamba run -p "$PWD/.envs/simulation" go2-ovd validate-camera
./.tools/micromamba run -p "$PWD/.envs/simulation" go2-ovd capture \
  --config configs/episodes/baseball_softball_v1.yaml \
  --output runs/baseball_softball_v1
```

For detector and evaluation commands:

```bash
HF_HOME="$PWD/.cache/huggingface" ./.venvs/perception/bin/go2-ovd detect \
  --config configs/detectors/grounding_dino_tiny.yaml \
  --frames runs/baseball_softball_v1 \
  --device mps

./.venvs/perception/bin/go2-ovd evaluate \
  --run runs/baseball_softball_v1 \
  --config configs/evaluation/default.yaml

./.venvs/perception/bin/go2-ovd report --run runs/baseball_softball_v1
```

Set `local_files_only: true` after bootstrap, or set `HF_HUB_OFFLINE=1`, to prove subsequent inference uses the cached immutable revision. The detector retains an archival 0.05-threshold JSONL and derives the declared 0.35 operating-point JSONL from it, so threshold sweeps do not require rerunning inference or cherry-picking a favorable threshold.

Run the full verification target with:

```bash
make verify
```

## Data contract

Each episode contains lossless PNG frames, compressed instance masks, `frames.jsonl`, `ground_truth.jsonl`, and `episode_manifest.json`. `T_world_camera` maps a homogeneous camera-frame point into world coordinates. Detector requests hash the raw RGB pixel bytes; frame records retain both that pixel hash and the PNG file hash. Boxes use clipped, half-open `xyxy` coordinates in the original image.

The future rejection interface is reserved in `src/go2_ovd/detectors/base.py`. A filter must write a new result file referencing the immutable raw detector output and configuration hash. It must never overwrite raw detections.

## Safety boundary

This is a simulation-only perception research system. It invokes no Unitree SDK command, ROS 2 actuator path, or physical robot. Passing MuJoCo tests does not establish safe contact-force realization, actuator protection, estimator correctness, watchdog behavior, safe stopping, transport timing, or physical Go2 safety. See `docs/real_go2_gate.md` before any hardware work.
