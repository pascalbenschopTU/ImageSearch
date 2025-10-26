#!/usr/bin/env python3
# patch_search_pipeline.py
import os
import csv
import glob
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# interactive UI
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Re-use your existing extractor.py
from extractor import ViTExtractor

# Optional FAISS
try:
    import faiss  # optional, for fast large-scale search
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False


# ---------------------------
# Utility helpers
# ---------------------------

def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def cast_floor(x: float) -> int:
    return int(math.floor(float(x)))


def _token_rc(idx: int, w_tokens: int) -> Tuple[int, int]:
    """Map a linear token index to (row, col) in the patch grid."""
    r = idx // w_tokens
    c = idx % w_tokens
    return int(r), int(c)


def _patch_box_in_load_space(
    r: int,
    c: int,
    patch_size: int,
    stride_hw: Tuple[int, int],
    load_h: int,
    load_w: int,
) -> Tuple[int, int, int, int]:
    """
    Bounding box (x0, y0, x1, y1) for the patch in the *resized* image
    used for descriptor extraction ("load space").
    """
    y0 = int(cast_floor(r * stride_hw[0]))
    x0 = int(cast_floor(c * stride_hw[1]))
    y1 = min(load_h, y0 + patch_size)
    x1 = min(load_w, x0 + patch_size)
    return x0, y0, x1, y1


# ---------------------------
# Core search engine
# ---------------------------

class PatchSearchEngine:
    """
    Build an index of ViT patch descriptors for a dataset of images and search
    for the most similar patches given one or more query descriptors.

    Index layout (saved as a single .pt file):
        {
          'meta': {
              'model_type': str,
              'stride': Tuple[int, int],
              'patch_size': int,
              'desc_dim': int
          },
          'features': FloatTensor [N_total_patches, D] (L2-normalized, float16),
          'image_ptrs': List[Tuple[int, int]]  # slice per image in `features`
          'paths': List[str],
          'orig_sizes': List[Tuple[int, int]],  # (H, W) of original images
          'load_sizes': List[Tuple[int, int]],  # (H, W) used for descriptors
          'patch_shapes': List[Tuple[int, int]],  # (H_tokens, W_tokens)
        }
    """

    def __init__(
        self,
        model_type: str = "dinov3_vitb16",
        stride: int = 4,
        device: str = "cuda",
        load_size: int = 224,
        use_faiss: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.extractor = ViTExtractor(model_type=model_type, stride=stride, device=device)
        self.model_type = model_type
        self.load_size = load_size
        self.patch_size: int = int(self.extractor.p)
        self.stride_hw: Tuple[int, int] = tuple(int(s) for s in self.extractor.stride)
        self.use_faiss = bool(use_faiss and _FAISS_AVAILABLE)

        # Filled after building/reading an index
        self.features: Optional[torch.Tensor] = None  # [N, D] float16, on CPU
        self.image_ptrs: List[Tuple[int, int]] = []
        self.paths: List[str] = []
        self.orig_sizes: List[Tuple[int, int]] = []
        self.load_sizes: List[Tuple[int, int]] = []
        self.patch_shapes: List[Tuple[int, int]] = []
        self.desc_dim: Optional[int] = None

        # FAISS handle
        self.faiss_index = None

    # -------- Indexing --------

    @torch.no_grad()
    def _image_to_tokens(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Compute per-patch descriptors for a single image.

        Returns:
            tokens: FloatTensor [T, D] L2-normalized (on CPU)
            orig_size: (H, W)
            load_size: (H, W)
            patch_shape: (H_tokens, W_tokens)
        """
        # Preprocess at the configured load_size but keep orig for mapping
        prep_tensor, pil = self.extractor.preprocess(image_path, (self.load_size, self.load_size))
        orig_size = pil.size[1], pil.size[0]  # PIL (W,H) -> (H,W)
        prep_tensor = prep_tensor.to(self.device)

        # Extract descriptors: [1, 1, t, D]
        desc = self.extractor.extract_descriptors(
            prep_tensor, layer=11, facet="token", bin=False, include_cls=False
        )

        # Pull shapes from extractor
        load_h, load_w = self.extractor.load_size
        h_tokens, w_tokens = self.extractor.num_patches

        # Flatten tokens and L2-normalize
        desc = desc.squeeze(0).squeeze(0)  # [t, D]
        desc = _l2_normalize(desc, dim=-1)

        return (
            desc.cpu().to(torch.float16),
            (orig_size[0], orig_size[1]),
            (load_h, load_w),
            (h_tokens, w_tokens),
        )

    @torch.no_grad()
    def build_index(
        self,
        images_dir: str,
        index_out: Optional[str] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        verbose: bool = True,
        show_progress: bool = True,
    ) -> Dict:
        """Walk a directory, compute per-patch descriptors, and assemble the index."""
        image_paths = [
            str(p)
            for p in sorted(Path(images_dir).rglob("*"))
            if p.suffix.lower() in extensions
        ]
        assert image_paths, f"No images found in {images_dir}."

        if verbose:
            print(f"Indexing {len(image_paths)} images with {self.model_type} @ load_size={self.load_size}...")

        features: List[torch.Tensor] = []
        image_ptrs: List[Tuple[int, int]] = []
        paths: List[str] = []
        orig_sizes: List[Tuple[int, int]] = []
        load_sizes: List[Tuple[int, int]] = []
        patch_shapes: List[Tuple[int, int]] = []

        offset = 0
        skipped = 0
        total_patches = 0
        iterator = tqdm(image_paths, total=len(image_paths), desc="Indexing images", dynamic_ncols=True) if show_progress else image_paths
        for img_path in iterator:
            try:
                tokens, orig_sz, load_sz, patch_shape = self._image_to_tokens(img_path)
            except Exception as e:
                if show_progress and hasattr(iterator, "set_postfix"):
                    skipped += 1
                    iterator.set_postfix(skipped=skipped)
                else:
                    print(f"[WARN] Skipping {img_path}: {e}")
                continue

            n_tok, d = tokens.shape
            if self.desc_dim is None:
                self.desc_dim = int(d)
            features.append(tokens)
            image_ptrs.append((offset, offset + n_tok))
            offset += n_tok

            paths.append(img_path)
            orig_sizes.append(orig_sz)
            load_sizes.append(load_sz)
            patch_shapes.append(patch_shape)

            total_patches += n_tok
            if show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(skipped=skipped, patches=total_patches)

        # Concatenate to a single tensor on CPU (float16 to save space)
        all_features = torch.cat(features, dim=0).contiguous().cpu().to(torch.float16)

        index = {
            "meta": {
                "model_type": self.model_type,
                "stride": self.stride_hw,
                "patch_size": self.patch_size,
                "desc_dim": self.desc_dim,
            },
            "features": all_features,
            "image_ptrs": image_ptrs,
            "paths": paths,
            "orig_sizes": orig_sizes,
            "load_sizes": load_sizes,
            "patch_shapes": patch_shapes,
        }

        # Persist
        if index_out is not None:
            os.makedirs(os.path.dirname(index_out) or ".", exist_ok=True)
            torch.save(index, index_out)
            if verbose:
                print(f"Saved index to {index_out} ({all_features.shape[0]} patches across {len(paths)} images).")

        # Cache in-memory
        self._attach_index(index)
        return index

    def _attach_index(self, index: Dict) -> None:
        self.features = index["features"]  # CPU float16
        self.image_ptrs = list(map(tuple, index["image_ptrs"]))
        self.paths = list(index["paths"])  # type: ignore
        self.orig_sizes = [tuple(x) for x in index["orig_sizes"]]
        self.load_sizes = [tuple(x) for x in index["load_sizes"]]
        self.patch_shapes = [tuple(x) for x in index["patch_shapes"]]
        self.desc_dim = int(index["meta"]["desc_dim"])  # type: ignore

        # Build FAISS if available/desired
        if self.use_faiss:
            if not _FAISS_AVAILABLE:
                print("[INFO] faiss not available; falling back to PyTorch search.")
            else:
                self.faiss_index = faiss.IndexFlatIP(self.desc_dim)
                feats32 = self.features.to(torch.float32).numpy()
                self.faiss_index.add(feats32)

    def load_index(self, index_path: str) -> Dict:
        index = torch.load(index_path, map_location="cpu")
        self._attach_index(index)
        return index

    def _image_id_from_global_idx(self, global_idx: int) -> int:
        # Binary search over image_ptrs
        lo, hi = 0, len(self.image_ptrs) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, end = self.image_ptrs[mid]
            if start <= global_idx < end:
                return mid
            elif global_idx < start:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(f"Global token idx {global_idx} out of range")

    # -------- Querying --------

    @torch.no_grad()
    def _load_query_descriptors(self, descriptor_paths: List[str]) -> torch.Tensor:
        """
        Load one or more saved descriptor .pt tensors and stack into [Q, D].
        Accepts tensors of shape [1,1,t,D] or [1,t,D] or [t,D] or [D].
        """
        q_list: List[torch.Tensor] = []
        for p in descriptor_paths:
            q = torch.load(p, map_location="cpu")
            # Normalize shapes
            if q.dim() == 1:                  # single descriptor vector [D]
                q = q.unsqueeze(0)            # -> [1, D]
            if q.dim() == 4:
                q = q.squeeze(0).squeeze(0)  # [t, D]
            elif q.dim() == 3:
                q = q.squeeze(0)
            elif q.dim() != 2:
                raise ValueError(f"Unsupported query tensor shape {tuple(q.shape)} from {p}")
            q = _l2_normalize(q, dim=-1)
            q_list.append(q.to(torch.float32))
        q_all = torch.cat(q_list, dim=0)  # [Q, D]
        return q_all
    
    def _load_descriptors_from_name(self, descriptordir: str, descriptorname: str) -> torch.Tensor:
        """
        Find all .pt files in descriptordir whose filename starts with descriptorname
        (e.g. descriptorname='drugs' -> drugs_1.pt, drugs_2.pt, ...),
        load them, normalize shapes, and concat into [Q, D].
        """
        pattern = os.path.join(descriptordir, f"*{descriptorname}*.pt")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No descriptors found for pattern {pattern}")

        return self._load_query_descriptors(paths)

    @torch.no_grad()
    def query(
        self,
        query_descriptors: torch.Tensor,
        topk_patches: int = 50,
        topk_images: int = 20,
        aggregate: str = "max",  # "max" or "mean"
        batch: int = 200000,     # chunk DB tokens to limit RAM
        show_progress: bool = True,
    ) -> Dict:
        """
        Search the index for the most similar patches/images.

        Returns a dict with two views:
            - 'top_patches': list of matches across the whole corpus
            - 'top_images': list aggregated per image
        """
        assert self.features is not None, "Index not loaded/built."
        Q, D = query_descriptors.shape
        assert D == self.desc_dim, f"Descriptor dim mismatch: query {D} vs index {self.desc_dim}"

        # Normalize the query again for safety
        q = _l2_normalize(query_descriptors.to(torch.float32), dim=-1)
        self._last_query = q  # stash for visualization

        # Compute similarity to every DB token
        if self.faiss_index is not None:
            sims, idxs = self.faiss_index.search(q.numpy(), topk_patches)  # [Q, topk]
            sims_t = torch.from_numpy(sims)
            idxs_t = torch.from_numpy(idxs)
            if aggregate == "max":
                best_sims, best_src = sims_t.max(dim=0)
                db_idxs = idxs_t[best_src, torch.arange(idxs_t.shape[1])]
            else:
                best_sims = sims_t.mean(dim=0)
                db_idxs = idxs_t[0]
        else:
            feats = self.features.to(torch.float32)  # [N, D] on CPU
            N = feats.shape[0]
            agg_scores = []
            agg_indices = []
            rng = tqdm(range(0, N, batch), total=math.ceil(N / batch),
                       desc="Searching patches", dynamic_ncols=True) if show_progress else range(0, N, batch)
            for start in rng:
                end = min(N, start + batch)
                chunk = feats[start:end]  # [n, D]
                sim = q @ chunk.t()       # [Q, n]
                s = sim.max(dim=0)[0] if aggregate == "max" else sim.mean(dim=0)
                agg_scores.append(s)
                agg_indices.append(torch.arange(start, end, dtype=torch.long))
                if show_progress and hasattr(rng, "set_postfix"):
                    rng.set_postfix(processed=end, total=N)
            scores = torch.cat(agg_scores, dim=0)   # [N]
            indices = torch.cat(agg_indices, dim=0) # [N]
            best_sims, top_idx = torch.topk(scores, k=min(topk_patches, scores.numel()))
            db_idxs = indices[top_idx]

        # Build patch-level results
        patch_results = []
        for sim_score, global_tok_idx in zip(best_sims.tolist(), db_idxs.tolist()):
            img_id = self._image_id_from_global_idx(global_tok_idx)
            start, end = self.image_ptrs[img_id]
            local_idx = global_tok_idx - start

            h_tokens, w_tokens = self.patch_shapes[img_id]
            r, c = _token_rc(local_idx, w_tokens)

            load_h, load_w = self.load_sizes[img_id]
            x0, y0, x1, y1 = _patch_box_in_load_space(r, c, self.patch_size, self.stride_hw, load_h, load_w)

            # Map to original image coordinates
            H_orig, W_orig = self.orig_sizes[img_id]
            scale_y = H_orig / float(load_h)
            scale_x = W_orig / float(load_w)
            x0o = int(round(x0 * scale_x)); y0o = int(round(y0 * scale_y))
            x1o = int(round(x1 * scale_x)); y1o = int(round(y1 * scale_y))

            patch_results.append({
                "image_id": img_id,
                "image_path": self.paths[img_id],
                "score": float(sim_score),
                "global_token": int(global_tok_idx),
                "local_token": int(local_idx),
                "token_rc": [int(r), int(c)],
                "box_load": [int(x0), int(y0), int(x1), int(y1)],
                "box_orig": [x0o, y0o, x1o, y1o],
            })

        # Aggregate per image (max over its tokens)
        per_image_best = {}
        for pr in patch_results:
            img_id = pr["image_id"]
            prev = per_image_best.get(img_id)
            if (prev is None) or (pr["score"] > prev["score"]):
                per_image_best[img_id] = pr

        # Rank images by their best patch score
        top_images = sorted(per_image_best.values(), key=lambda x: -x["score"])[:topk_images]

        return {
            "top_patches": patch_results,
            "top_images": top_images,
        }

    # -------- Convenience I/O --------
    def save_top_images_csv(
        self,
        results: Dict,
        csv_path: str,
    ) -> None:
        """
        Save only the per-image best matches as CSV.

        CSV columns:
            Image_filename,patch_confidence
        where:
            Image_filename = basename of the matched image
            patch_confidence = similarity score of that image's best patch
        """
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image_filename", "patch_confidence"])
            for r in results["top_images"]:
                img_name = os.path.basename(r["image_path"])
                conf = float(r["score"])
                writer.writerow([img_name, conf])

        print(f"Saved CSV to {csv_path}")

    @torch.no_grad()
    def save_heatmaps(
        self,
        results: Dict,
        out_dir: str,
        aggregate: str = "max",
        alpha: float = 0.45,
        blur_ksize: int = 0,
    ) -> None:
        """
        For each image in results['top_images'], compute a dense patch similarity map
        and overlay it as a heatmap on the original image.
        """
        os.makedirs(out_dir, exist_ok=True)
        q = getattr(self, "_last_query", None)
        assert q is not None, "No cached query found. Call engine.query(...) first."

        for rank, r in enumerate(results["top_images"]):
            img_id = r["image_id"]
            img_path = r["image_path"]
            (h_tokens, w_tokens) = self.patch_shapes[img_id]
            (H_orig, W_orig) = self.orig_sizes[img_id]
            start, end = self.image_ptrs[img_id]

            feats = self.features[start:end].to(torch.float32)  # [T, D]
            # similarity per token
            sim = (q @ feats.t())  # [Q, T]
            s = sim.max(dim=0)[0] if aggregate == "max" else sim.mean(dim=0)  # [T]
            s = s.view(h_tokens, w_tokens).cpu().numpy()

            # normalize 0..1
            s = s - s.min()
            if s.max() > 0:
                s = s / s.max()

            # upsample to original image size
            heat = cv2.resize(s, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)
            if blur_ksize and blur_ksize > 0:
                heat = cv2.GaussianBlur(heat, (blur_ksize, blur_ksize), 0)

            heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"[WARN] Could not read {img_path}")
                continue

            if img_bgr.dtype != np.uint8:
                img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
            if heat_color.shape[:2] != img_bgr.shape[:2]:
                H, W = img_bgr.shape[:2]
                heat_color = cv2.resize(heat_color, (W, H), interpolation=cv2.INTER_CUBIC)

            overlay = cv2.addWeighted(img_bgr, 1.0, heat_color, alpha, 0.0)

            out_path = os.path.join(out_dir, f"{rank:03d}_{os.path.basename(img_path)}")
            cv2.imwrite(out_path, overlay)


# ---------------------------
# Descriptor creation (image/video)
# ---------------------------

def get_frame_from_video(video_path: str) -> str:
    """
    Opens the video, shows a slider to select a frame, and returns a temporary PNG path.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        raise SystemExit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read the first frame.")
        raise SystemExit(1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)
    im_disp = ax.imshow(frame_rgb)
    ax.set_title("Use the slider to choose a frame, then click 'Select Frame'")

    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    frame_slider = Slider(ax_slider, 'Frame', 0, total_frames - 1, valinit=0, valstep=1)

    selected_frame = None

    def update(val):
        frame_idx = int(frame_slider.val)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, fr = cap.read()
        if ok:
            fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            im_disp.set_data(fr_rgb)
            fig.canvas.draw_idle()

    frame_slider.on_changed(update)

    ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
    select_button = Button(ax_button, 'Select Frame')

    def select(event):
        nonlocal selected_frame
        frame_idx = int(frame_slider.val)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, fr = cap.read()
        if ok:
            selected_frame = fr.copy()
        plt.close(fig)

    select_button.on_clicked(select)
    plt.show()
    cap.release()

    if selected_frame is None:
        print("No frame selected. Exiting.")
        raise SystemExit(1)

    selected_frame_rgb = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)
    temp_image_path = "temp_extracted_frame.png"
    # save without implicit resizing—we’ll resize in extractor.preprocess
    plt.imsave(temp_image_path, selected_frame_rgb)
    return temp_image_path


@torch.no_grad()
def interactive_descriptors(
    image_path: str,
    descriptorname: str,
    out_dir: str = "output/descriptors/",
    model_type: str = "dinov3_vitb16",
    load_size: int = 224,
    stride: int = 4,
    facet: str = "token",
    layer: int = 11,
    device = "cpu",
) -> List[str]:
    """
    Interactively click patches on an image and save their descriptors (.pt) one-by-one.
    Returns the list of saved descriptor paths.
    """
    # # Device
    # device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Build extractor (for DINOv3, stride is internally fixed to patch size)
    extractor = ViTExtractor(model_type=model_type, stride=stride, device=device)

    # Always pass an int load_size to extractor.preprocess
    image_batch, image_pil = extractor.preprocess(image_path, load_size=int(load_size))

    # Prefer facet='token'; extractor handles DINOv3 tokens (drops CLS+registers)
    descr = extractor.extract_descriptors(
        image_batch.to(device), layer=layer, facet=facet, bin=False, include_cls=False
    )
    # Shapes and geometry
    num_patches = extractor.num_patches            # (H_tokens, W_tokens)
    load_h, load_w = extractor.load_size          # resized image size used for descriptors
    patch_size = int(extractor.p)                 # e.g., 16 for DINOv3
    stride_hw = tuple(int(s) for s in extractor.stride)  # (sh, sw)

    # Output dir
    os.makedirs(out_dir, exist_ok=True)
    reference_image_path = os.path.join(out_dir, f"{descriptorname}_reference.png")

    # Display the resized image we actually indexed to keep coordinates aligned
    fig, ax = plt.subplots(figsize=(10, 6))
    interactive_title = "Click to save a descriptor (right click to exit)"
    fig.suptitle(interactive_title)
    ax.imshow(image_pil)
    radius = patch_size // 2
    descriptor_count = 1

    saved = []

    # Click loop
    while True:
        pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
        if len(pts) != 1:  # right-click or ESC ends
            break

        # pts in the coordinate system of `image_pil` (load_size x load_size)
        y_px, x_px = int(pts[0, 1]), int(pts[0, 0])

        # Convert pixel to token grid using *stride per axis*
        sh, sw = stride_hw
        ht, wt = num_patches
        patch_y = y_px // sh
        patch_x = x_px // sw

        if not (0 <= patch_x < wt and 0 <= patch_y < ht):
            print(f"Clicked outside valid token grid: ({patch_x}, {patch_y}) of ({wt}, {ht})")
            continue

        raveled_idx = patch_y * wt + patch_x
        # descr: [1, 1, t, D]
        point_descriptor = descr[0, 0, raveled_idx]  # [D]
        print(f"Selected patch ({patch_x}, {patch_y}) → idx {raveled_idx}, descriptor shape {tuple(point_descriptor.shape)}")

        # Visual cue
        center_x = patch_x * sw + patch_size // 2
        center_y = patch_y * sh + patch_size // 2
        patch_circle = plt.Circle((center_x, center_y), radius, color=(1, 0, 0, 0.75))
        ax.add_patch(patch_circle)
        fig.canvas.draw()

        # Save descriptor
        out_path = os.path.join(out_dir, f"{descriptorname}_{descriptor_count}.pt")
        torch.save(point_descriptor.cpu(), out_path)
        saved.append(out_path)
        print(f"Saved descriptor → {out_path}")
        descriptor_count += 1

        # Update and save the reference overlay
        fig.suptitle(f"Descriptors saved as '{descriptorname}_*.pt'")
        fig.savefig(reference_image_path)
        print(f"Reference overlay saved → {reference_image_path}")
        fig.suptitle(interactive_title)

    return saved


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Patch-level visual search with interactive descriptor creation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- index
    p_index = sub.add_parser("index", help="Build an index over an image directory")
    p_index.add_argument("images_dir", type=str, help="Directory with images to index")
    p_index.add_argument("index_out", type=str, help="Output .pt index file")
    p_index.add_argument("--model_type", type=str, default="dinov3_vitb16")
    p_index.add_argument("--stride", type=int, default=4, help="Ignored for DINOv3")
    p_index.add_argument("--device", type=str, default="cuda")
    p_index.add_argument("--load_size", type=int, default=224)
    p_index.add_argument("--no_faiss", action="store_true", help="Disable FAISS even if installed")

    # ---- extract descriptor (interactive)
    p_desc = sub.add_parser("extract", help="Interactively create one or more descriptors from an image or video frame")
    io_group = p_desc.add_mutually_exclusive_group(required=True)
    io_group.add_argument("-i", "--image", type=str, help="Template image to pick descriptors from")
    io_group.add_argument("-v", "--video", type=str, help="Video file to pick a frame from")
    p_desc.add_argument("-o", "--out", type=str, default="output/descriptors/", help="Where to store descriptors")
    p_desc.add_argument("-n", "--descriptorname", type=str, required=True, help="Base descriptor name; suffix is added")
    p_desc.add_argument("--model_type", type=str, default="dinov3_vitb16")
    p_desc.add_argument("--load_size", type=int, default=224)
    p_desc.add_argument("--stride", type=int, default=4, help="Ignored for DINOv3")
    p_desc.add_argument("--facet", type=str, default="token", choices=["token", "key", "query", "value"])
    p_desc.add_argument("--layer", type=int, default=11)
    p_desc.add_argument("--device", type=str, default="cuda")

    # ---- search
    p_search = sub.add_parser("search", help="Search an existing index using saved descriptor .pt files")
    p_search.add_argument("index_path", type=str, help="Path to .pt index file")
    p_search.add_argument("-n", "--descriptorname", type=str, required=True, help="Base descriptor name; suffix is added")
    p_search.add_argument("-d", "--descriptordir", type=str, default="output/descriptors/", help="Where to store descriptors")
    p_search.add_argument("--topk_patches", type=int, default=500)
    p_search.add_argument("--topk_images", type=int, default=20)
    p_search.add_argument("--aggregate", type=str, default="max", choices=["max", "mean"])
    p_search.add_argument("--results_csv", type=str, default="search_results.csv")
    p_search.add_argument("--save_crops_dir", type=str, default=None)
    p_search.add_argument("--heatmaps", action="store_true", help="Export heatmap overlays for top images.")
    p_search.add_argument("--heatmap_dir", type=str, default="heatmaps", help="Directory to write heatmap overlays.")
    p_search.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha.")
    p_search.add_argument("--crop_size", type=int, default=None)
    p_search.add_argument("--device", type=str, default="cuda")
    p_search.add_argument("--use_faiss", action="store_true", help="Use FAISS (requires faiss)")

    # ---- pipeline: index -> extract -> search
    p_pipe = sub.add_parser("pipeline", help="Do everything: index -> interactive descriptors -> search")
    p_pipe.add_argument("images_dir", type=str, help="Directory with images to index")
    p_pipe.add_argument("index_out", type=str, help="Output .pt index file")
    p_pipe.add_argument("-n", "--descriptorname", type=str, required=True, help="Base descriptor name for saved .pt files")
    io_group2 = p_pipe.add_mutually_exclusive_group(required=True)
    io_group2.add_argument("-i", "--image", type=str, help="Template image to pick descriptors from")
    io_group2.add_argument("-v", "--video", type=str, help="Video file to pick a frame from")

    # shared knobs
    p_pipe.add_argument("-o", "--out", type=str, default="output/descriptors/", help="Where to store descriptors")
    p_pipe.add_argument("--results_csv", type=str, default="search_results.csv")
    p_pipe.add_argument("--save_crops_dir", type=str, default=None)
    p_pipe.add_argument("--heatmaps", action="store_true")
    p_pipe.add_argument("--heatmap_dir", type=str, default="heatmaps")
    p_pipe.add_argument("--alpha", type=float, default=0.45)
    p_pipe.add_argument("--crop_size", type=int, default=None)

    # model/extractor knobs
    p_pipe.add_argument("--model_type", type=str, default="dinov3_vitb16")
    p_pipe.add_argument("--stride", type=int, default=4)
    p_pipe.add_argument("--device", type=str, default="cuda")
    p_pipe.add_argument("--load_size", type=int, default=224)
    p_pipe.add_argument("--facet", type=str, default="token", choices=["token", "key", "query", "value"])
    p_pipe.add_argument("--layer", type=int, default=11)
    p_pipe.add_argument("--use_faiss", action="store_true")
    p_pipe.add_argument("--aggregate", type=str, default="max", choices=["max", "mean"])
    p_pipe.add_argument("--topk_patches", type=int, default=500)
    p_pipe.add_argument("--topk_images", type=int, default=20)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.cmd == "index":
        engine = PatchSearchEngine(
            model_type=args.model_type,
            stride=args.stride,
            device=args.device,
            load_size=args.load_size,
            use_faiss=not args.no_faiss,
        )
        engine.build_index(images_dir=args.images_dir, index_out=args.index_out)

    elif args.cmd == "extract":
        if args.video:
            print(f"Opening video: {args.video}")
            input_image_path = get_frame_from_video(args.video)
        else:
            input_image_path = args.image

        with torch.no_grad():
            interactive_descriptors(
                image_path=input_image_path,
                descriptorname=args.descriptorname,
                out_dir=args.out,
                model_type=args.model_type,
                load_size=args.load_size,
                stride=args.stride,
                facet=args.facet,
                layer=args.layer,
                device=args.device,
            )

    elif args.cmd == "search":
        engine = PatchSearchEngine(device=args.device, use_faiss=args.use_faiss)
        engine.load_index(args.index_path)
        # q = engine._load_query_descriptors(args.descriptors)
        q = engine._load_descriptors_from_name(args.descriptordir, args.descriptorname)
        if q.shape[1] != engine.desc_dim:
            raise ValueError(f"Query dim {q.shape[1]} doesn't match index dim {engine.desc_dim}.")
        results = engine.query(
            q,
            topk_patches=args.topk_patches,
            topk_images=args.topk_images,
            aggregate=args.aggregate
        )
        if getattr(args, "results_csv", None):
            csv_path = (args.results_csv if args.results_csv.lower().endswith(".csv") else args.results_csv + ".csv").replace(".csv", f"_{args.descriptorname}.csv")
            engine.save_top_images_csv(results, csv_path=csv_path)

        if args.heatmaps:
            engine.save_heatmaps(
                results,
                out_dir=args.heatmap_dir,
                aggregate=args.aggregate,
                alpha=args.alpha,
            )

    elif args.cmd == "pipeline":
        # 1) index
        engine = PatchSearchEngine(
            model_type=args.model_type,
            stride=args.stride,
            device=args.device,
            load_size=args.load_size,
            use_faiss=args.use_faiss,
        )
        engine.build_index(images_dir=args.images_dir, index_out=args.index_out)

        # 2) interactive descriptors from image or video
        if args.video:
            print(f"Opening video: {args.video}")
            input_image_path = get_frame_from_video(args.video)
        else:
            input_image_path = args.image

        with torch.no_grad():
            saved_paths = interactive_descriptors(
                image_path=input_image_path,
                descriptorname=args.descriptorname,
                out_dir=args.out,
                model_type=args.model_type,
                load_size=args.load_size,
                stride=args.stride,
                facet=args.facet,
                layer=args.layer,
                device=args.device,
            )

        if not saved_paths:
            raise SystemExit("No descriptors saved; aborting search.")

        # 3) search
        q = engine._load_query_descriptors(saved_paths)
        if q.shape[1] != engine.desc_dim:
            raise ValueError(f"Query dim {q.shape[1]} doesn't match index dim {engine.desc_dim}.")
        results = engine.query(
            q,
            topk_patches=args.topk_patches,
            topk_images=args.topk_images,
            aggregate=args.aggregate
        )
        if getattr(args, "results_csv", None):
            csv_path = (args.results_csv if args.results_csv.lower().endswith(".csv") else args.results_csv + ".csv").replace(".csv", f"_{args.descriptorname}.csv")
            engine.save_top_images_csv(results, csv_path=csv_path)


        if args.heatmaps:
            engine.save_heatmaps(
                results,
                out_dir=args.heatmap_dir,
                aggregate=args.aggregate,
                alpha=args.alpha,
            )


if __name__ == "__main__":
    main()
