#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Directories for persisting positive/negative embeddings across sessions.
# Relative to this script so they follow the project regardless of cwd.
_SCRIPT_DIR = Path(__file__).parent
_EMBED_DIR_POS = _SCRIPT_DIR / "demo_embeddings" / "positive"
_EMBED_DIR_NEG = _SCRIPT_DIR / "demo_embeddings" / "negative"

MODEL_ALIASES = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vith14": "facebook/dinov3-vith14-pretrain-lvd1689m",
    "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}

# Gaussian blur kernel applied after bicubic upscale for segmentation-like soft edges.
# Mirrors the blur_ksize pattern in image_search.PatchSearchEngine.save_heatmaps.
_BLUR_KSIZE = 17

# Set to cv2.COLORMAP_TURBO (or JET fallback) after cv2 is imported.
_COLORMAP = None


def load_runtime_dependencies(requested_device: str) -> None:
    global cv2, np, torch, F, Image, AutoImageProcessor, AutoModel, _COLORMAP

    architecture_issue = detect_architecture_issue()
    if architecture_issue is not None and requested_device in {"auto", "mps"}:
        print(f"Warning: {architecture_issue}", file=sys.stderr)

    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
    except Exception as exc:
        raise SystemExit(
            "Missing runtime dependency. Install missing packages with: "
            "pip install transformers timm safetensors opencv-python pillow numpy"
        ) from exc

    _COLORMAP = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)

    # torch.compiler.is_compiling() was added in 2.3; the fast image processor
    # calls it unconditionally.  We never use torch.compile() so False is correct.
    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False

    if _version_tuple(torch.__version__) < (2, 2):
        raise SystemExit(
            f"Installed torch=={torch.__version__}, but this stack requires torch>=2.2. "
            "Upgrade torch and torchvision together, e.g.:\n"
            "  pip install --upgrade torch torchvision"
        )

    try:
        import torchvision
    except Exception as exc:
        raise SystemExit(
            "Missing or broken torchvision. Install a version matching torch, e.g.:\n"
            "  pip install --upgrade torch torchvision"
        ) from exc

    torch_major_minor = _version_tuple(torch.__version__)[:2]
    torchvision_major_minor = _version_tuple(torchvision.__version__)[:2]
    if torch_major_minor >= (2, 4) and torchvision_major_minor < (0, 19):
        raise SystemExit(
            f"Installed torchvision=={torchvision.__version__}; upgrade it with torch:\n"
            "  pip install --upgrade torch torchvision"
        )

    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:
        raise SystemExit(
            "Missing or incompatible transformers. Install/update it with:\n"
            "  pip install --upgrade transformers timm safetensors"
        ) from exc


def _version_tuple(version: str) -> Tuple[int, ...]:
    clean = version.split("+", 1)[0]
    parts = []
    for part in clean.split("."):
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def detect_architecture_issue() -> Optional[str]:
    process_machine = platform.machine()
    try:
        host_machine = subprocess.check_output(
            ["uname", "-m"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        host_machine = process_machine

    if host_machine == "arm64" and process_machine == "x86_64":
        return (
            "Python is x86_64 (Rosetta) on an arm64 Mac. "
            "MPS may work via Rosetta but native arm64 Python gives better performance.\n"
            "To fix: install Miniforge3 arm64 and recreate the environment."
        )
    return None


def choose_device(requested: str) -> "torch.device":
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_ref(model_ref: str) -> str:
    if model_ref in MODEL_ALIASES:
        return MODEL_ALIASES[model_ref]

    path = Path(model_ref).expanduser()
    if not path.exists():
        return model_ref

    if (path / "config.json").exists():
        return str(path)

    refs_main = path / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = path / "snapshots" / revision
        if (snapshot / "config.json").exists():
            return str(snapshot)

    snapshots = sorted((path / "snapshots").glob("*/config.json"))
    if snapshots:
        return str(snapshots[-1].parent)

    raise FileNotFoundError(
        f"Could not find a Hugging Face model snapshot under {path}"
    )


def l2_normalize(features: "torch.Tensor") -> "torch.Tensor":
    return F.normalize(features, p=2, dim=-1, eps=1e-8)


# ---------------------------------------------------------------------------
# Similarity primitives — module-level so they are independently testable
# ---------------------------------------------------------------------------

def compute_per_patch_max_similarity(
    frame_tokens: "torch.Tensor",   # shape [T, D], L2-normalized patch embeddings of the frame
    query_tokens: "torch.Tensor",   # shape [Q, D], L2-normalized embeddings of selected query patches
) -> "torch.Tensor":                # shape [T], one similarity score per frame patch
    """
    For each of the T frame patch tokens, compute the maximum cosine similarity
    across all Q positive query patch tokens.

    A high score means "this frame patch closely resembles at least one of the
    selected query patches."  Keeping all Q query patches separate (rather than
    averaging them first) means that selecting two visually different objects as
    positive correctly produces high scores near both of them in the frame.
    """
    similarity_matrix = frame_tokens @ query_tokens.T   # [T, Q]
    per_frame_patch_max, _ = similarity_matrix.max(dim=1)
    return per_frame_patch_max                          # [T]


def compute_mean_query_similarity(
    frame_tokens: "torch.Tensor",   # shape [T, D], L2-normalized
    query_tokens: "torch.Tensor",   # shape [Q, D], L2-normalized
) -> "torch.Tensor":                # shape [T], one similarity score per frame patch
    """
    Average all Q query patch tokens into a single "holistic object descriptor",
    L2-normalize it, then compute cosine similarity against every frame patch.

    This captures the scenario where the object appears smaller or farther away
    than at query time.  A single frame patch that shows the whole (compressed)
    object will resemble the averaged appearance of the object's query patches
    more than it resembles any individual close-up query patch.
    """
    mean_query_descriptor = query_tokens.mean(dim=0, keepdim=True)   # [1, D]
    mean_query_descriptor = l2_normalize(mean_query_descriptor)       # [1, D]
    similarity_scores = frame_tokens @ mean_query_descriptor.T        # [T, 1]
    return similarity_scores.squeeze(1)                               # [T]


def pool_token_grid_spatially(
    frame_tokens_flat: "torch.Tensor",  # shape [H*W, D], row-major order, L2-normalized
    grid_h: int,                        # number of patch rows in the native token grid
    grid_w: int,                        # number of patch columns in the native token grid
    pool_scale: int,                    # spatial pooling factor (e.g. 2 → 2×2 average pool)
) -> "Tuple[torch.Tensor, int, int]":  # (pooled_tokens [h_out*w_out, D], h_out, w_out)
    """
    Reshape the flat token sequence back into a 2D spatial grid, average-pool every
    pool_scale×pool_scale neighbourhood of adjacent patch tokens into one super-token,
    then L2-normalize the super-tokens and return them flattened again.

    The resulting super-tokens each represent a larger spatial region of the frame.
    Matching the original query patches against these super-tokens detects the object
    when it appears larger or closer than at query time, where its appearance is spread
    across multiple native patches that individually may not match well on their own.

    Returns:
        pooled_tokens  — [h_out * w_out, D] L2-normalized super-token embeddings
        h_out          — number of super-token rows  (= grid_h // pool_scale)
        w_out          — number of super-token columns (= grid_w // pool_scale)
    """
    token_grid = frame_tokens_flat.reshape(grid_h, grid_w, -1)   # [H, W, D]

    # Crop so that H and W are exactly divisible by pool_scale before reshaping
    cropped_h = (grid_h // pool_scale) * pool_scale
    cropped_w = (grid_w // pool_scale) * pool_scale
    token_grid_cropped = token_grid[:cropped_h, :cropped_w, :]    # [H', W', D]

    h_out = cropped_h // pool_scale
    w_out = cropped_w // pool_scale

    # Rearrange into non-overlapping pool_scale×pool_scale blocks, then average each block
    blocks = token_grid_cropped.reshape(h_out, pool_scale, w_out, pool_scale, -1)
    pooled_grid = blocks.mean(dim=(1, 3))                          # [h_out, w_out, D]

    pooled_flat = pooled_grid.reshape(h_out * w_out, -1)          # [h_out*w_out, D]
    pooled_flat = l2_normalize(pooled_flat)   # re-normalize after averaging changes the norm
    return pooled_flat, h_out, w_out


def compute_multiscale_similarity(
    frame_tokens: "torch.Tensor",   # shape [T, D], L2-normalized (T = grid_h * grid_w)
    query_tokens: "torch.Tensor",   # shape [Q, D], L2-normalized
    grid_h: int,
    grid_w: int,
    pool_scales: List[int],         # spatial pooling factors, e.g. [2, 4]
) -> "np.ndarray":                  # shape [grid_h, grid_w], float32
    """
    Compute three families of similarity maps from a single set of frame tokens
    and average them into one combined map at native [grid_h, grid_w] resolution.

    Family 1 — fine-grained per-patch max (always included):
        Each frame patch is compared individually against every query patch.
        The maximum similarity across query patches is kept per frame patch.
        Best for: object at similar scale and distance as when the query was selected.

    Family 2 — mean-query holistic match (always included):
        All query patches are averaged into one descriptor before matching.
        That single descriptor is compared against each native-resolution frame patch.
        Best for: object appearing farther away (fewer patches) than at query time —
        one frame patch shows the whole compressed object, resembling the query mean.

    Family 3 — spatially pooled frame × per-patch max (one map per entry in pool_scales):
        Native frame tokens are average-pooled into coarser super-tokens at each scale.
        Each super-token is matched against all query patches (max over queries).
        The resulting coarse map is upsampled back to [grid_h, grid_w].
        Best for: object appearing closer (more patches) than at query time —
        a pooled neighbourhood aggregates its spread features.

    All maps contain raw cosine similarity values, so their scale is compatible with
    the pos_thresh and neg_thresh trackbars that the caller applies afterward.
    """
    maps_to_average: List["np.ndarray"] = []

    # --- Family 1: fine-grained per-patch max ---
    fine_scores = compute_per_patch_max_similarity(frame_tokens, query_tokens)
    fine_map = fine_scores.reshape(grid_h, grid_w).cpu().float().numpy()
    maps_to_average.append(fine_map)

    # --- Family 2: mean-query holistic match ---
    mean_scores = compute_mean_query_similarity(frame_tokens, query_tokens)
    mean_map = mean_scores.reshape(grid_h, grid_w).cpu().float().numpy()
    maps_to_average.append(mean_map)

    # --- Family 3: spatially pooled frame tokens, one map per requested scale ---
    for scale in pool_scales:
        if scale >= min(grid_h, grid_w):
            # Pooling at this scale would collapse most of the grid — skip it
            continue

        pooled_tokens, h_out, w_out = pool_token_grid_spatially(
            frame_tokens, grid_h, grid_w, scale
        )
        pooled_scores = compute_per_patch_max_similarity(pooled_tokens, query_tokens)
        pooled_map_coarse = pooled_scores.reshape(h_out, w_out).cpu().float().numpy()

        # Upsample to native resolution so all maps can be averaged element-wise
        pooled_map_upsampled = cv2.resize(
            pooled_map_coarse, (grid_w, grid_h), interpolation=cv2.INTER_LINEAR
        )
        maps_to_average.append(pooled_map_upsampled)

    # Stack all maps and average element-wise — equal weight to every family
    combined_map = np.mean(np.stack(maps_to_average, axis=0), axis=0)
    return combined_map.astype(np.float32)


class DinoTokenMatcher:
    def __init__(
        self,
        model_ref: str,
        device: "torch.device",
        load_size: int,
        pool_scales: List[int],
        query_zoom_levels: List[float],
    ) -> None:
        resolved = resolve_model_ref(model_ref)
        local_only = Path(resolved).exists()

        print(f"Loading model: {resolved}")
        self.processor = AutoImageProcessor.from_pretrained(
            resolved, local_files_only=local_only
        )
        self.model = AutoModel.from_pretrained(
            resolved, local_files_only=local_only
        )
        self.model.eval().to(device)

        self.device = device
        self.load_size = int(load_size)
        self.pool_scales = pool_scales
        self.query_zoom_levels = query_zoom_levels
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        self.num_register_tokens = int(
            getattr(self.model.config, "num_register_tokens", 0)
        )

        self.query_tokens: Optional["torch.Tensor"] = None
        self.query_roi: Optional[Tuple[int, int, int, int]] = None
        self.negative_tokens: Optional["torch.Tensor"] = None
        self.negative_roi: Optional[Tuple[int, int, int, int]] = None

    def frame_tokens(
        self, frame_bgr: "np.ndarray"
    ) -> Tuple["torch.Tensor", Tuple[int, int]]:
        """Run the model on one BGR frame; return (tokens [T,D] L2-normed, (grid_h, grid_w))."""
        with torch.no_grad():
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                size={"height": self.load_size, "width": self.load_size},
            )
            pixel_values = inputs["pixel_values"].to(self.device)
            outputs = self.model(pixel_values)
            tokens = outputs.last_hidden_state[:, 1 + self.num_register_tokens:, :]
            tokens = l2_normalize(tokens.squeeze(0).to(torch.float32))
            input_h, input_w = pixel_values.shape[-2:]
            return tokens, (input_h // self.patch_size, input_w // self.patch_size)

    def _extract_roi_tokens(
        self,
        frame_bgr: "np.ndarray",
        roi: Tuple[int, int, int, int],
    ) -> Tuple["torch.Tensor", int]:
        """Run model on frame, return the subset of L2-normed tokens that fall inside roi."""
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            raise ValueError("ROI must have positive width and height")

        tokens, (grid_h, grid_w) = self.frame_tokens(frame_bgr)
        frame_h, frame_w = frame_bgr.shape[:2]

        x0 = x * self.load_size / frame_w
        y0 = y * self.load_size / frame_h
        x1 = (x + w) * self.load_size / frame_w
        y1 = (y + h) * self.load_size / frame_h

        selected = []
        for row in range(grid_h):
            cy = (row + 0.5) * self.patch_size
            for col in range(grid_w):
                cx = (col + 0.5) * self.patch_size
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    selected.append(row * grid_w + col)

        if not selected:
            # ROI too small to cover any patch centre — pick the nearest one
            cx0 = (x0 + x1) * 0.5
            cy0 = (y0 + y1) * 0.5
            col = int(np.clip(cx0 // self.patch_size, 0, grid_w - 1))
            row = int(np.clip(cy0 // self.patch_size, 0, grid_h - 1))
            selected.append(row * grid_w + col)

        return tokens[selected].detach(), len(selected)

    def extract_roi_crop_tokens(
        self,
        frame_bgr: "np.ndarray",
        roi: Tuple[int, int, int, int],
        zoom_levels: List[float],
    ) -> "torch.Tensor":
        """
        Build multi-scale "GT" tokens for the selected object by running the model
        on context windows of increasing size centred on the ROI.

        At zoom 1.0 the context window is exactly the ROI crop — the object fills
        the entire model input, giving maximum-detail tokens.

        At zoom 0.5 the context window is twice the ROI dimensions, so after the
        model resizes it to load_size the object occupies half the grid.  This is
        exactly what the model sees when the real object is twice as far from the
        camera — same appearance, same surrounding context, no artificial background.

        At zoom 0.25 the context window is four times the ROI, simulating four times
        the distance, and so on.

        Only tokens whose patch centre falls inside the ROI region are kept;
        surrounding-context tokens are discarded so they do not pollute the query bank.

        Returns a [N_total, D] tensor of L2-normalised tokens across all zoom levels.
        """
        x_roi, y_roi, w_roi, h_roi = roi
        frame_h, frame_w = frame_bgr.shape[:2]
        cx = x_roi + w_roi // 2   # ROI centre in frame pixels
        cy = y_roi + h_roi // 2

        all_zoom_tokens: List["torch.Tensor"] = []

        for zoom in zoom_levels:
            # Context window: 1/zoom times the ROI dimensions, centred on ROI centre
            ctx_w = int(w_roi / zoom)
            ctx_h = int(h_roi / zoom)

            # Context window corners (may extend beyond frame — clamped below)
            ctx_x0 = cx - ctx_w // 2
            ctx_y0 = cy - ctx_h // 2
            ctx_x1 = ctx_x0 + ctx_w
            ctx_y1 = ctx_y0 + ctx_h

            # Clamp to frame boundaries
            ctx_x0c = max(0, ctx_x0)
            ctx_y0c = max(0, ctx_y0)
            ctx_x1c = min(frame_w, ctx_x1)
            ctx_y1c = min(frame_h, ctx_y1)

            ctx_crop = frame_bgr[ctx_y0c:ctx_y1c, ctx_x0c:ctx_x1c]
            if ctx_crop.size == 0:
                continue

            # Run the model on the context crop (processor handles resize to load_size)
            tokens, (grid_h, grid_w) = self.frame_tokens(ctx_crop)

            # Map ROI bounds into the clamped context crop's coordinate space,
            # then into the model's load_size space so we can select the right tokens.
            actual_ctx_w = ctx_x1c - ctx_x0c
            actual_ctx_h = ctx_y1c - ctx_y0c

            roi_in_ctx_x0 = x_roi - ctx_x0c
            roi_in_ctx_y0 = y_roi - ctx_y0c
            roi_in_ctx_x1 = roi_in_ctx_x0 + w_roi
            roi_in_ctx_y1 = roi_in_ctx_y0 + h_roi

            roi_in_load_x0 = roi_in_ctx_x0 * self.load_size / actual_ctx_w
            roi_in_load_y0 = roi_in_ctx_y0 * self.load_size / actual_ctx_h
            roi_in_load_x1 = roi_in_ctx_x1 * self.load_size / actual_ctx_w
            roi_in_load_y1 = roi_in_ctx_y1 * self.load_size / actual_ctx_h

            # Keep only the patch tokens whose centre lies inside the ROI region
            roi_token_indices = []
            for row in range(grid_h):
                patch_centre_y = (row + 0.5) * self.patch_size
                for col in range(grid_w):
                    patch_centre_x = (col + 0.5) * self.patch_size
                    if (roi_in_load_x0 <= patch_centre_x <= roi_in_load_x1
                            and roi_in_load_y0 <= patch_centre_y <= roi_in_load_y1):
                        roi_token_indices.append(row * grid_w + col)

            if roi_token_indices:
                all_zoom_tokens.append(tokens[roi_token_indices].detach())

        if not all_zoom_tokens:
            return torch.zeros(
                (0, self.model.config.hidden_size), dtype=torch.float32
            )
        return torch.cat(all_zoom_tokens, dim=0)

    def set_query_from_roi(
        self,
        frame_bgr: "np.ndarray",
        roi: Tuple[int, int, int, int],
        embed_dir: Optional[Path] = None,
    ) -> int:
        """
        Extract positive query tokens from roi and accumulate them with any previously
        selected positive tokens (does NOT overwrite earlier selections).

        Two sets of tokens are combined per selection:
          - Native-scale tokens from _extract_roi_tokens: the ROI patches as seen in
            the full frame at the current model resolution.
          - Zoom-level tokens from extract_roi_crop_tokens: the same object at
            progressively smaller apparent sizes, built by expanding the context window
            and re-running the model.  These let the matcher recognise the object at
            greater distances without any additional per-frame cost.

        If embed_dir is provided the combined new tokens are saved as a timestamped
        .pt file for persistence across sessions.

        Returns the number of native-scale tokens added (not the running total).
        """
        native_tokens, count = self._extract_roi_tokens(frame_bgr, roi)

        zoom_tokens = self.extract_roi_crop_tokens(frame_bgr, roi, self.query_zoom_levels)

        if zoom_tokens.shape[0] > 0:
            all_new_tokens = torch.cat([native_tokens, zoom_tokens], dim=0)
        else:
            all_new_tokens = native_tokens

        if self.query_tokens is not None:
            self.query_tokens = torch.cat([self.query_tokens, all_new_tokens], dim=0)
        else:
            self.query_tokens = all_new_tokens
        self.query_roi = roi

        if embed_dir is not None:
            save_embedding(all_new_tokens, embed_dir)
        return count

    def set_negative_from_roi(
        self,
        frame_bgr: "np.ndarray",
        roi: Tuple[int, int, int, int],
        embed_dir: Optional[Path] = None,
    ) -> int:
        """
        Extract negative query tokens from roi and accumulate them with any previously
        selected negative tokens (does NOT overwrite earlier selections).

        Applies the same native + zoom-level extraction as set_query_from_roi so that
        the negative suppression also works across different object distances.

        If embed_dir is provided the combined new tokens are saved as a timestamped
        .pt file for persistence across sessions.

        Returns the number of native-scale tokens added (not the running total).
        """
        native_tokens, count = self._extract_roi_tokens(frame_bgr, roi)

        zoom_tokens = self.extract_roi_crop_tokens(frame_bgr, roi, self.query_zoom_levels)

        if zoom_tokens.shape[0] > 0:
            all_new_tokens = torch.cat([native_tokens, zoom_tokens], dim=0)
        else:
            all_new_tokens = native_tokens

        if self.negative_tokens is not None:
            self.negative_tokens = torch.cat([self.negative_tokens, all_new_tokens], dim=0)
        else:
            self.negative_tokens = all_new_tokens
        self.negative_roi = roi

        if embed_dir is not None:
            save_embedding(all_new_tokens, embed_dir)
        return count

    def match_frame(
        self,
        frame_bgr: "np.ndarray",
    ) -> Tuple["np.ndarray", Optional["np.ndarray"], Tuple[int, int]]:
        """
        Run the model on one frame and return separate positive and negative
        similarity grids at native patch resolution.

        Both grids are produced by compute_multiscale_similarity, which averages
        three families of maps: fine per-patch max, holistic mean-query, and
        spatially-pooled frame regions.  Keeping pos and neg separate lets the
        main loop cheaply recompute the combined display whenever the sliders change
        without re-running the model.
        """
        if self.query_tokens is None:
            raise RuntimeError("Select a positive ROI before matching frames")

        with torch.no_grad():
            frame_tokens, (grid_h, grid_w) = self.frame_tokens(frame_bgr)

            pos_grid = compute_multiscale_similarity(
                frame_tokens, self.query_tokens, grid_h, grid_w, self.pool_scales
            )

            neg_grid: Optional["np.ndarray"] = None
            if self.negative_tokens is not None:
                neg_grid = compute_multiscale_similarity(
                    frame_tokens, self.negative_tokens, grid_h, grid_w, self.pool_scales
                )

        return pos_grid, neg_grid, (grid_h, grid_w)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def compute_display_grid(
    pos_grid: "np.ndarray",
    neg_grid: Optional["np.ndarray"],
    pos_thresh: float,
    neg_thresh: float,
    neg_weight: float,
) -> "np.ndarray":
    """
    Threshold each similarity map independently (values below threshold → 0),
    then subtract the scaled negative map from the positive map.

    Thresholding before subtraction means weak positive evidence is zeroed out
    before the negative penalty is applied, and vice-versa — matching the
    user's intent: each slider cuts off its own map independently.
    """
    pos = np.clip(pos_grid - pos_thresh, 0.0, None)
    if neg_grid is not None and neg_weight > 0.0:
        neg = np.clip(neg_grid - neg_thresh, 0.0, None)
        return pos - neg_weight * neg
    return pos


def overlay_segmentation_heatmap(
    frame_bgr: "np.ndarray",
    display_grid: "np.ndarray",
    alpha_max: float,
) -> "np.ndarray":
    """
    Upscale the patch-resolution grid to frame size (bicubic), smooth edges with
    a Gaussian blur, then blend using per-pixel alpha proportional to score.

    Negative / suppressed regions (score ≤ 0) are fully transparent — they show
    the original frame unchanged, giving a segmentation-mask feel rather than a
    full-frame rainbow.  Mirrors the resize + blur + colormap approach in
    image_search.PatchSearchEngine.save_heatmaps.
    """
    H, W = frame_bgr.shape[:2]

    heat = np.clip(display_grid, 0.0, None)
    if heat.max() < 1e-6:
        return frame_bgr.copy()
    heat = heat / heat.max()

    heat_up = cv2.resize(heat, (W, H), cv2.INTER_CUBIC)
    if _BLUR_KSIZE > 1:
        ksize = (_BLUR_KSIZE // 2) * 2 + 1  # ensure odd
        heat_up = cv2.GaussianBlur(heat_up, (ksize, ksize), 0)
    heat_up = np.clip(heat_up, 0.0, 1.0)

    heat_u8 = (heat_up * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, _COLORMAP)

    alpha_map = heat_up[:, :, np.newaxis] * alpha_max
    return (
        frame_bgr.astype(np.float32) * (1.0 - alpha_map)
        + heat_color.astype(np.float32) * alpha_map
    ).astype(np.uint8)


def draw_status(
    frame_bgr: "np.ndarray",
    text: str,
    line: int = 0,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    y = 26 + line * 24
    cv2.putText(frame_bgr, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Embedding persistence helpers
# ---------------------------------------------------------------------------

def save_embedding(tokens: "torch.Tensor", embed_dir: Path) -> Path:
    """
    Save a token embedding tensor to a timestamped .pt file inside embed_dir.

    Each call creates a new file (never overwrites), so successive selections
    accumulate on disk.  The filename encodes the save time so files sort
    chronologically and can be inspected easily.

    Returns the path of the newly written file.
    """
    embed_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = embed_dir / f"{timestamp}.pt"
    torch.save(tokens.cpu(), out_path)
    return out_path


def load_all_embeddings(embed_dir: Path) -> Optional["torch.Tensor"]:
    """
    Load every .pt file in embed_dir and concatenate them into a single tensor.

    Files are processed in sorted (chronological) order.  Returns None when the
    directory does not exist or contains no .pt files.
    """
    if not embed_dir.exists():
        return None
    pt_files = sorted(embed_dir.glob("*.pt"))
    if not pt_files:
        return None
    loaded_tensors = [torch.load(p, weights_only=True) for p in pt_files]
    return torch.cat(loaded_tensors, dim=0)


def clear_embeddings(embed_dir: Path) -> int:
    """
    Delete every .pt file in embed_dir.

    Returns the number of files deleted so the caller can report it to the user.
    Does nothing (returns 0) when the directory does not exist.
    """
    if not embed_dir.exists():
        return 0
    pt_files = list(embed_dir.glob("*.pt"))
    for file_path in pt_files:
        file_path.unlink()
    return len(pt_files)


# ---------------------------------------------------------------------------
# Interactive ROI selector with grid overlay
# ---------------------------------------------------------------------------

def select_roi_with_grid_overlay(
    window_name: str,
    frame_bgr: "np.ndarray",
    grid_h: int,
    grid_w: int,
) -> Tuple[int, int, int, int]:
    """
    Custom ROI selector that overlays the patch grid on the frame and highlights
    selected patches in green as the user drags.

    Controls: click-drag to draw, ENTER/SPACE to confirm, c/ESC to cancel.
    Returns (x, y, w, h) in image coordinates, or (0, 0, 0, 0) on cancel.
    """
    H, W = frame_bgr.shape[:2]
    cell_w = W / grid_w
    cell_h = H / grid_h

    sel: dict = {"x0": -1, "y0": -1, "x1": -1, "y1": -1, "dragging": False, "cancel": False}

    def mouse_cb(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            sel["x0"] = sel["x1"] = x
            sel["y0"] = sel["y1"] = y
            sel["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and sel["dragging"]:
            sel["x1"] = x
            sel["y1"] = y
        elif event == cv2.EVENT_LBUTTONUP:
            sel["x1"] = x
            sel["y1"] = y
            sel["dragging"] = False

    cv2.setMouseCallback(window_name, mouse_cb)

    while True:
        img = frame_bgr.copy()

        has_box = sel["x0"] >= 0
        if has_box:
            rx  = min(sel["x0"], sel["x1"])
            ry  = min(sel["y0"], sel["y1"])
            rx2 = max(sel["x0"], sel["x1"])
            ry2 = max(sel["y0"], sel["y1"])

            # Semi-transparent green fill for patches whose centre falls in the box
            patch_layer = img.copy()
            for row in range(grid_h):
                cy = (row + 0.5) * cell_h
                for col in range(grid_w):
                    cx = (col + 0.5) * cell_w
                    if rx <= cx <= rx2 and ry <= cy <= ry2:
                        cv2.rectangle(
                            patch_layer,
                            (int(col * cell_w), int(row * cell_h)),
                            (int((col + 1) * cell_w), int((row + 1) * cell_h)),
                            (0, 220, 0), -1,
                        )
            img = cv2.addWeighted(img, 0.65, patch_layer, 0.35, 0)

        # Grid lines drawn on top so they stay visible over the green fill
        for col in range(1, grid_w):
            gx = int(col * cell_w)
            cv2.line(img, (gx, 0), (gx, H), (180, 180, 180), 1)
        for row in range(1, grid_h):
            gy = int(row * cell_h)
            cv2.line(img, (0, gy), (W, gy), (180, 180, 180), 1)

        # Selection rectangle
        if has_box:
            cv2.rectangle(img, (rx, ry), (rx2, ry2), (0, 255, 255), 2)

        draw_status(img, "Drag to select patches   ENTER/SPACE confirm   c/ESC cancel", 0)
        cv2.imshow(window_name, img)

        key = cv2.waitKey(16) & 0xFF
        if key in (13, ord(" ")):   # ENTER or SPACE
            break
        elif key in (27, ord("c")):  # ESC or c
            sel["cancel"] = True
            break

    cv2.setMouseCallback(window_name, lambda *args: None)  # detach callback

    if sel["cancel"] or sel["x0"] < 0:
        return (0, 0, 0, 0)

    rx  = min(sel["x0"], sel["x1"])
    ry  = min(sel["y0"], sel["y1"])
    rw  = abs(sel["x1"] - sel["x0"])
    rh  = abs(sel["y1"] - sel["y0"])
    return (rx, ry, rw, rh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track / segment a webcam object using DINOv3 patch-token similarity."
    )
    parser.add_argument("--model", default="./dinov3", help="Local model dir or HF model id")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280, help="Requested webcam width")
    parser.add_argument("--height", type=int, default=720, help="Requested webcam height")
    parser.add_argument("--load-size", type=int, default=224, help="Square DINO input size")
    parser.add_argument(
        "--query-zoom-levels",
        type=float,
        nargs="*",
        default=[1.0, 0.5, 0.25],
        metavar="Z",
        help=(
            "Context-window zoom levels used when building the GT query token bank. "
            "At 1.0 the ROI crop fills the model input (max detail). "
            "At 0.5 the context window is 2× the ROI so the object appears at half scale, "
            "matching how it looks when twice as far away. "
            "Pass no values to disable zoom augmentation."
        ),
    )
    parser.add_argument(
        "--pool-scales",
        type=int,
        nargs="*",
        default=[2, 4],
        metavar="S",
        help=(
            "Spatial pooling factors for multi-scale matching. "
            "E.g. --pool-scales 2 4 (default) adds 2×2 and 4×4 pooled maps. "
            "Pass no values (--pool-scales) to use only fine-grained matching."
        ),
    )
    parser.add_argument(
        "--heatmap-threshold",
        type=float,
        default=0.72,
        help="Initial value for the Pos threshold trackbar (0–1)",
    )
    parser.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.60,
        help="Maximum segmentation overlay opacity",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help=(
            "Number of patch tokens per side, overrides --load-size "
            "(e.g. 14 = 224px, 20 = 320px, 28 = 448px for a 16px patch model)"
        ),
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=1,
        help="Run DINO every N webcam frames (1 = every frame)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    load_runtime_dependencies(args.device)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        print("Tip: export PYTORCH_ENABLE_MPS_FALLBACK=1 if an MPS op is missing.")

    matcher = DinoTokenMatcher(
        model_ref=args.model,
        device=device,
        load_size=args.load_size,
        pool_scales=args.pool_scales,
        query_zoom_levels=args.query_zoom_levels,
    )

    if args.grid_size is not None:
        matcher.load_size = args.grid_size * matcher.patch_size
        print(f"Grid: {args.grid_size}×{args.grid_size} patches (load_size={matcher.load_size})")
    grid_cells = matcher.load_size // matcher.patch_size

    # Load any embeddings saved in a previous session
    saved_pos = load_all_embeddings(_EMBED_DIR_POS)
    if saved_pos is not None:
        matcher.query_tokens = saved_pos.to(device)
        print(f"Loaded {len(saved_pos)} positive tokens from {_EMBED_DIR_POS}")
    saved_neg = load_all_embeddings(_EMBED_DIR_NEG)
    if saved_neg is not None:
        matcher.negative_tokens = saved_neg.to(device)
        print(f"Loaded {len(saved_neg)} negative tokens from {_EMBED_DIR_NEG}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "DINOv3 webcam token tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Trackbars — integer handles mapped to floats in the loop:
    #   Pos thresh %   : 0–99  →  0.00–0.99  cosine-similarity floor for the positive map
    #   Neg thresh %   : 0–99  →  0.00–0.99  cosine-similarity floor for the negative map
    #   Neg weight x10 : 0–30  →  0.0–3.0    scale on the negative map before subtraction
    pos_thresh_init = min(int(args.heatmap_threshold * 100), 99)
    cv2.createTrackbar("Pos thresh %",   window_name, pos_thresh_init, 99, lambda _: None)
    cv2.createTrackbar("Neg thresh %",   window_name, 50,              99, lambda _: None)
    cv2.createTrackbar("Neg weight x10", window_name, 10,              30, lambda _: None)

    paused = False
    show_heatmap = True
    held_frame: Optional["np.ndarray"] = None
    # pos_grid and neg_grid are cached from the last model inference so that
    # slider changes can update the display instantly without re-running the model.
    last_pos_grid: Optional["np.ndarray"] = None
    last_neg_grid: Optional["np.ndarray"] = None
    frame_index = 0

    while True:
        if not paused or held_frame is None:
            ok, frame = cap.read()
            if not ok:
                break
            held_frame = frame
        else:
            frame = held_frame.copy()

        display = frame.copy()
        has_query = matcher.query_tokens is not None

        # Model inference — only when live
        if has_query and not paused and frame_index % max(args.process_every, 1) == 0:
            last_pos_grid, last_neg_grid, _ = matcher.match_frame(frame)

        # Read sliders every frame (cheap — just trackbar reads + numpy ops)
        pos_thresh = cv2.getTrackbarPos("Pos thresh %",   window_name) / 100.0
        neg_thresh = cv2.getTrackbarPos("Neg thresh %",   window_name) / 100.0
        neg_weight = cv2.getTrackbarPos("Neg weight x10", window_name) / 10.0

        # Segmentation overlay — recomputed from cached grids + current slider values
        if has_query and show_heatmap and last_pos_grid is not None:
            display_grid = compute_display_grid(
                last_pos_grid, last_neg_grid, pos_thresh, neg_thresh, neg_weight
            )
            display = overlay_segmentation_heatmap(display, display_grid, args.heatmap_alpha)

        # ROI outlines when paused (yellow = positive query, red = negative)
        if paused:
            if matcher.query_roi is not None:
                x, y, w, h = matcher.query_roi
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
            if matcher.negative_roi is not None:
                x, y, w, h = matcher.negative_roi
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Status text
        state = "paused" if paused else "live"
        has_neg = matcher.negative_tokens is not None
        if has_query and has_neg:
            n_pos = len(matcher.query_tokens)
            n_neg = len(matcher.negative_tokens)
            query_label = f"pos({n_pos}) + neg({n_neg})"
        elif has_query:
            query_label = f"pos({len(matcher.query_tokens)}) set"
        else:
            query_label = "press s to select object"
        draw_status(display, f"{state}  |  {query_label}", 0)
        draw_status(display, "s pos  n neg  space pause  h overlay  c clear  q quit", 1)
        if last_pos_grid is not None:
            neg_info = f"  neg_max={last_neg_grid.max():.2f}" if last_neg_grid is not None else ""
            draw_status(display, f"pos_max={last_pos_grid.max():.2f}{neg_info}", 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key in (ord(" "), ord("p")):
            paused = not paused

        elif key == ord("h"):
            show_heatmap = not show_heatmap

        elif key == ord("c"):
            matcher.query_tokens = None
            matcher.query_roi = None
            matcher.negative_tokens = None
            matcher.negative_roi = None
            last_pos_grid = None
            last_neg_grid = None
            n_pos_deleted = clear_embeddings(_EMBED_DIR_POS)
            n_neg_deleted = clear_embeddings(_EMBED_DIR_NEG)
            print(f"Cleared all queries (deleted {n_pos_deleted} positive, {n_neg_deleted} negative files).")

        elif key == ord("s"):
            paused = True
            selection_frame = held_frame.copy()
            roi = select_roi_with_grid_overlay(window_name, selection_frame, grid_cells, grid_cells)
            if roi[2] > 0 and roi[3] > 0:
                count = matcher.set_query_from_roi(selection_frame, roi, embed_dir=_EMBED_DIR_POS)
                total_pos = len(matcher.query_tokens)
                # Refresh both grids immediately so the overlay shows while still paused.
                last_pos_grid, last_neg_grid, _ = matcher.match_frame(selection_frame)
                print(f"Added positive ROI {roi}: +{count} tokens (total positive: {total_pos})")
            else:
                print("Positive ROI selection cancelled.")

        elif key == ord("n"):
            paused = True
            selection_frame = held_frame.copy()
            roi = select_roi_with_grid_overlay(window_name, selection_frame, grid_cells, grid_cells)
            if roi[2] > 0 and roi[3] > 0:
                count = matcher.set_negative_from_roi(selection_frame, roi, embed_dir=_EMBED_DIR_NEG)
                total_neg = len(matcher.negative_tokens)
                # Refresh grids so negative suppression is visible immediately.
                if matcher.query_tokens is not None:
                    last_pos_grid, last_neg_grid, _ = matcher.match_frame(selection_frame)
                print(f"Added negative ROI {roi}: +{count} tokens (total negative: {total_neg})")
            else:
                print("Negative ROI selection cancelled.")

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()