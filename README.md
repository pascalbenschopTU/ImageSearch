# Image search with DINO

A script that can be used to:
- extract local visual descriptors (“patch descriptors”) from images or a frame in a video,
- build an index of these descriptors for an entire image folder,
- interactively pick one or more descriptors from a reference image,
- search for the most similar patches / images in the index,
- optionally generate heatmaps and crops to visualize where matches were found.

## Understanding patches and results

### How many patches per image?

When the model processes an image, it first resizes it to a square of size `load_size x load_size`.  
By default this script uses `--load_size 224`.

The ViT doesn't embed the whole image at once. It splits the resized image into a uniform grid of patches.  
For DINOv3 ViTs like `dinov3_vitb16`, the patch size is 16×16 pixels. That means:

- number of patches along width  = 224 / 16 = 14  
- number of patches along height = 224 / 16 = 14  
- total patches per image        = 14 * 14 = 196

Important:
- This `196` is what you’re indexing per image.
- If you increase `--load_size`, you get more patches per image (quadratically).
  - Example: `load_size = 336`
    - 336 / 16 = 21
    - total patches = 21 * 21 = 441
  - So bigger `load_size` = denser spatial resolution = heavier index (more tokens in memory / FAISS).

This patch count is exactly what shows up as `extractor.num_patches` and is what the script saves into the index as `patch_shapes` for each image.

### How many results do you get back?

The search returns two ranked views:

1. **Patch-level matches** (`results["top_patches"]`):
   - Each entry is: “this specific patch (x,y region) from image X looks similar to your query descriptor.”
   - The parameter `--topk_patches` controls how many of these patch hits you keep globally.
   - Example: `--topk_patches 50`  
     → You get up to 50 best patch matches across the entire database, sorted by similarity score.

   If you increase `--topk_patches`, you will:
   - see more individual hit boxes from more images (and sometimes multiple hits on the same image),
   - produce more crops if you enabled `--save_crops_dir`,
   - make bigger JSON output.

2. **Image-level matches** (`results["top_images"]`):
   - For each indexed image, we look at *all* its patches and keep only that image’s single best patch score.
   - Then we sort images by that best score and take the top images.
   - The parameter `--topk_images` controls how many distinct images you keep.
   - Example: `--topk_images 20`  
     → You get up to 20 unique images, each with metadata about where the best match is inside that image.

So:
- `topk_patches` = “show me this many *individual local regions* that match.”
- `topk_images`  = “show me this many *unique images* that contain good matches.”

They’re independent. You can ask for a ton of patches (like 200) but only keep the top 10 images, or vice versa.

---

## Requirements

```

transformers (at least4.56.0.dev0)
torch
torchvision
opencv-python
matplotlib
timm
numpy

````

## image search

Test data can be downloaded at https://www.kaggle.com/datasets/apollo2506/image-search-engine


Full pipeline: build index -> interactively pick descriptors -> search -> (optional) heatmaps

```
python image_search.py pipeline {images folder root} {search database filename}.pt -n {descriptor name} -i {image file for extracting descriptor} -o {descriptor folder} --heatmaps --device mps/cuda/cpu

python image_search.py pipeline data/ImageSearch/ test.pt -n test_desc -i data/ImageSearch/snow/snow_000001.png --heatmaps --device mps
```

Just build an index over a folder of images
```
python image_search.py index {images folder root} {search database filename}.pt --device mps/cuda/cpu

python image_search.py index data/ImageSearch/ test.pt --device mps
```

Just extract / save one or more descriptors from an image (or video frame)
```
python image_search.py extract -i {image file for extracting descriptor} -n {descriptor name} -o {descriptor folder} --device mps

python image_search.py extract -i data/ImageSearch/snow/snow_000001.png -n test_desc -o desc_folder --device mps
```

Just search an existing index with one or more saved descriptors
```
python image_search.py search {search database filename}.pt -n {descriptor name} -d {descriptor folder} --heatmaps --device mps

python image_search.py search test.pt -n test_desc --heatmaps --device mps
```


### MPS device error

If you get this error:

```text
NotImplementedError: The operator 'aten::upsample_bicubic2d.out' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
```

Then just do this before executing the script:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
