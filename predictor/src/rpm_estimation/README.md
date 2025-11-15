# Rotating Object Extraction (ROO)

This module implements clustering algorithms for detecting rotating objects (propellers) from event-based camera data.

## Overview

The ROO module uses a combination of heatmap-based centroid initialization and K-means clustering to identify propeller regions in drone footage. The key innovation is **blob filling**, which creates solid, continuous regions without holes.

## Blob Filling

### What is Blob Filling?

Blob filling transforms sparse cluster assignments into solid, filled regions that:
- **Have no interior holes**: All pixels within the blob outline are labeled
- **Stretch to their outlines**: Regions extend to their natural boundaries
- **Connect nearby points**: Small gaps between cluster points are closed

### How It Works

The blob filling process consists of three steps:

1. **Morphological Closing**: Connects nearby cluster points and closes small gaps
   ```python
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
   mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
   ```

2. **Contour Detection**: Finds the outer boundary of each cluster
   ```python
   contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   ```

3. **Region Filling**: Fills the interior of each contour
   ```python
   cv2.fillPoly(filled_map, [contour], label)
   ```

### Configuration

Blob filling is controlled by three parameters in `src/roo/config.py`:

- **`FILL_BLOBS`** (bool, default: `True`): Enable/disable blob filling
- **`BLOB_KERNEL_SIZE`** (int, default: `5`): Size of morphological kernel
  - Larger values = more aggressive gap closing
  - Smaller values = tighter fit to original points
- **`USE_CONVEX_HULL`** (bool, default: `False`): Use convex hulls instead of contours
  - `True`: Maximum extent, convex shape (no concavities)
  - `False`: Natural shape, follows actual cluster boundary

### Visual Comparison

**Before Blob Filling** (sparse clusters with holes):
```
Label Map:
  0  0  -1  -1  0
  0  -1  -1  0  0
  -1  -1  0  0  0
```

**After Blob Filling** (solid regions):
```
Label Map:
  0  0  0  0  0
  0  0  0  0  0
  0  0  0  0  0
```

## Usage

### Basic Usage

```python
from src.roo.rotating_object_extraction import find_clusters

# Find clusters with default blob filling
labels = find_clusters(x_coords, y_coords, drone_crop_coords=crop)
```

### Custom Blob Parameters

```python
# Use convex hulls for maximum extent
labels = find_clusters(
    x_coords, y_coords,
    fill_blobs=True,
    blob_kernel_size=7,  # More aggressive filling
    use_convex_hull=True,  # Convex shapes
)

# Disable blob filling (original sparse clusters)
labels = find_clusters(
    x_coords, y_coords,
    fill_blobs=False,
)
```

### Extracting Blob Contours

```python
from src.roo.kmeans import extract_blob_contours

# Get contour outlines for each blob
contours_by_label = extract_blob_contours(labels)

# Access contours for specific label
for contour in contours_by_label[0]:
    # Each contour is a numpy array of (x, y) points
    print(f"Contour has {len(contour)} points")
```

## Visualization

The module produces two types of visualizations:

1. **`heatmap.png`**: Event density heatmap with centroids marked in red
2. **`labels_with_contours.png`**: Colored blob regions with white contour outlines

Both are saved to the `logs/` directory.

## Algorithm Pipeline

```
Event Data (x, y coordinates)
    ↓
Create Heatmap (find_heatmap)
    ↓
Initialize Centroids (initialize_centroids_from_heatmap)
    ↓
K-Means Clustering (kmeans_clustering)
    ↓
Remove Outliers (remove_outliers)
    ↓
Convert to Label Map (events_labels_to_heatmap_labels)
    ↓
Fill Blobs (fill_blob_regions) ← NEW STEP
    ↓
Extract Contours (extract_blob_contours)
    ↓
Filled Label Map + Blob Contours
```

## Key Functions

### `fill_blob_regions(label_map, kernel_size, use_convex_hull)`
Creates filled blob regions without holes.

**Parameters:**
- `label_map`: Sparse cluster assignments
- `kernel_size`: Morphological kernel size (default: 5)
- `use_convex_hull`: Use convex hulls vs. natural contours (default: False)

**Returns:** Filled label map with solid regions

### `extract_blob_contours(label_map)`
Extracts contour outlines for each blob.

**Returns:** Dictionary mapping `label → list of contours`

### `locate_centroids(scene, k, fill_blobs, blob_kernel_size, use_convex_hull)`
Main clustering function with blob filling support.

**Parameters:**
- `scene`: Heatmap array
- `k`: Number of clusters (default: 4 for 4 propellers)
- `fill_blobs`: Enable blob filling (default: True)
- `blob_kernel_size`: Kernel size (default: 5)
- `use_convex_hull`: Use convex hulls (default: False)

**Returns:** `(label_map, centers_x, centers_y)`

## Configuration Reference

File: `src/roo/config.py`

```python
# Original clustering parameters
EPSILON = 0.5                        # Centroid selection threshold
MAX_ITERATIONS = 100                 # K-means max iterations
CONVERGENCE_THRESHOLD = 1e-3         # Convergence threshold
OUTLIER_THRESHOLD_MULTIPLIER = 5     # Outlier detection threshold

# Blob filling parameters (NEW)
FILL_BLOBS = True                    # Enable blob filling
BLOB_KERNEL_SIZE = 5                 # Morphological kernel size
USE_CONVEX_HULL = False              # Use convex hulls
```

## Benefits of Blob Filling

1. **Robust Detection**: Solid regions are more reliable for downstream processing
2. **Better Tracking**: Continuous regions improve frame-to-frame tracking
3. **Accurate Boundaries**: Morphological operations respect natural cluster shapes
4. **No Holes**: Interior pixels are always labeled, preventing fragmentation
5. **Configurable**: Adjustable parameters for different propeller sizes and speeds

## Integration with EVM

Blob filling works synergistically with Eulerian Video Magnification (EVM):
- EVM enhances propeller motion → more visible events
- Blob filling creates solid propeller regions → better clustering
- Together: Robust propeller detection and tracking

## File Structure

```
src/roo/
├── config.py                        # Configuration parameters
├── kmeans.py                        # Clustering + blob filling
├── rotating_object_extraction.py   # Main interface + visualization
└── README.md                        # This file
```

## Performance Considerations

- Blob filling adds minimal overhead (~5-10ms per frame)
- Morphological operations are highly optimized in OpenCV
- Convex hulls are slightly faster but less accurate than contour filling
- Can be disabled with `fill_blobs=False` for comparison

## Examples

### Example 1: Default Configuration
```python
# Uses config defaults (FILL_BLOBS=True, BLOB_KERNEL_SIZE=5)
labels = find_clusters(x, y, drone_crop_coords=crop)
```

### Example 2: Aggressive Filling for Small Propellers
```python
# Larger kernel to connect distant propeller points
labels = find_clusters(
    x, y,
    blob_kernel_size=9,
    use_convex_hull=False,
)
```

### Example 3: Convex Hulls for Fast-Moving Propellers
```python
# Convex hulls provide maximum extent
labels = find_clusters(
    x, y,
    use_convex_hull=True,
)
```

### Example 4: No Blob Filling (Original Behavior)
```python
# Compare results with/without filling
labels_sparse = find_clusters(x, y, fill_blobs=False)
labels_filled = find_clusters(x, y, fill_blobs=True)
```
