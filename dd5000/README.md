# Drone Detector 5000 (Challenge 1)

Drone Detector 5000 is based on a simple event loop (`src/main.py`).

- Events are taken into the system whenever they're created (`src/evio_lib/`)
- At a configurable interval, temporally concatenate the events into a frame (`src/main.py`)
- The frame is fed to a fine-tuned YOLO model (`src/yolo.py`)
    - YOLO detects if the event contains a drone (`src/yolo.py`)
        - If not, discard frame
        - If yet, continue
- YOLO gives us a bounding box containing the drone (`src/yolo.py`)
- We use KMeans to detect clusters of high frequency events inside of the bounding box. These will be the drone's propellers. (`src/kmeans.py`)
    - KMeans is customized to be intialized via customized centroids (`src/kmeans.py`)
    - KMeans auto-adjusts its' chosen `k` (`src/kmeans.py`)
- Once we have `k` clusters of high event frequency, we process each of these clusters separately (`src/main.py`)
- We once again run KMeans to detect blades of the rotor, with similar principles. (`src/kmeans.py`)
- We use the blade count for each rotor to compute RPM (`src/rpm_estimator.py`)

## Standalone RPM Calculation

For testing and validation purposes, `src/pure_rpm_calculations.py` provides a standalone implementation that calculates RPM directly from event data using a hardcoded ROI (Region of Interest). 

**How it works**: Events are grouped into time windows (1ms default), and for each window, the number of "on" events (positive polarity) within the ROI are counted using boolean masking (`extract_roi_intensity()`). This creates a time-series signal of event activity. After collecting 100 samples, FFT analysis (`np.fft.rfft()`) converts the signal to frequency domain, identifying the dominant frequency peak which represents blade passage rate. RPM is calculated as `(frequency_hz * 60) / blade_count`, converting blade passages per minute to full rotations per minute.


## Technical achievements
- In our experiments, the fine-tuned YOLO model could detect drones frame-by-frame, running on a normal Macbook, in only 8ms. This is a significant finding, as even the Sensofusion challenge providers did not believe this could be achieved.
    - With a YOLO-based drone detection system, it's possible to label enemy drones accurately in real time. It is possible to teach YOLO models to detect different drone types.
- We completely implemented the KNN propeller detection concept from [this publication](https://arxiv.org/pdf/2209.02205). 


## Development philosophy
- We utilized a mild functional development philosophy. The aim was to create composable modules that could be called on their lonesome using `uv -m`. (`src/main.py`, `src/kmeans.py`, `src/yolo.py`)
- We did some software testing where it made sense in the interest of time. The k-means implementation needed testing to verify that it works (`tests/test_kmeans.py`), and we optimized our final product via `tests/test_non_functional.py`, that automatically ran the prediction on all benchmark datasets.
- We implemented a profiler [TODO]
