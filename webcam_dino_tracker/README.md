# Webcam DINO Token Tracker

Interactive proof of concept for using DINOv3 patch-token similarity to track an object from a webcam.

The demo does not need any image dataset. It captures a webcam frame, lets you select a region, stores the DINO patch tokens inside that region as the query, and then matches those tokens against each new webcam frame.

## Setup

For MPS on Apple Silicon, the Python environment must be native `arm64`. If `conda info` shows `platform : osx-64`, you are in an Intel/Rosetta conda and pip will not find modern MPS-capable `torch` wheels.

Recommended: create a fresh native arm64 environment from an arm64 Miniforge/Miniconda install, then run from `ImageSearch/`:

```bash
conda create -n video_features_arm python=3.11 -y
conda activate video_features_arm
export PYTORCH_ENABLE_MPS_FALLBACK=1
pip install --upgrade torch torchvision
pip install --upgrade transformers timm safetensors
pip install opencv-python pillow numpy
```

Important:
- Recent `transformers` builds used by DINOv3 disable PyTorch when `torch<2.4` is installed.
- If pip only lists torch versions up to `2.2.2`, the active Python is likely `x86_64`/`osx-64`; switch to native arm64 Python instead of forcing that install.
- If you keep an old torch environment for other work, pin NumPy with `pip install "numpy<2"` to avoid NumPy 2.x compiled-extension warnings.

## Run

```bash
cd /Users/pbenschop1/repos/ImageSearch
python webcam_dino_tracker/demo.py --device mps --model ./dinov3
```

The `--model ./dinov3` path resolves the local Hugging Face snapshot folder in this repo. You can also pass a resolved snapshot path or a model id such as `facebook/dinov3-vitb16-pretrain-lvd1689m`.

## Controls

- `s`: select an object ROI from the current frame.
- `space` or `p`: pause/resume the webcam view.
- `c`: clear the current query.
- `h`: toggle heatmap overlay.
- `q` or `esc`: quit.

## Useful knobs

```bash
python webcam_dino_tracker/demo.py \
  --device mps \
  --model ./dinov3 \
  --load-size 224 \
  --camera 0 \
  --heatmap-threshold 0.72 \
  --query-aggregate max
```

- `--load-size 224` is the fastest default and gives a 14x14 token grid for 16px patches.
- `--load-size 336` gives denser matching but is slower.
- `--query-aggregate max` matches any selected query token; `mean` uses one averaged object descriptor.
- `--heatmap-threshold` controls how much of the similarity map becomes the tracking box.

## Notes

This is not a production tracker. There is no motion model or temporal re-identification; it simply recomputes DINO patch tokens on each processed frame and finds where the selected query tokens are most similar.
