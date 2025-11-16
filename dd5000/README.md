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

**How it works:**
- Events are grouped into 1ms time windows
- "On" events (positive polarity) within the ROI are counted per window using boolean masking
- Creates a time-series signal of event activity (100 samples collected)
- FFT analysis (`np.fft.rfft()`) identifies the dominant frequency peak (blade passage rate)
- RPM = `(frequency_hz * 60) / blade_count` (converts blade passages/min to rotations/min)


## Technical achievements
- In our experiments, the fine-tuned YOLO model could detect drones frame-by-frame, running on a normal Macbook, in only 8ms. This is a significant finding, as even the Sensofusion challenge providers did not believe this could be achieved.
    - With a YOLO-based drone detection system, it's possible to label enemy drones accurately in real time. It is possible to teach YOLO models to detect different drone types.
- We completely implemented the KNN propeller detection concept from [this publication](https://arxiv.org/pdf/2209.02205). 


## Development philosophy
- We utilized a mild functional development philosophy. The aim was to create composable modules that could be called on their lonesome using `uv -m`. (`src/main.py`, `src/kmeans.py`, `src/yolo.py`)
- We did some software testing where it made sense in the interest of time. The k-means implementation needed testing to verify that it works (`tests/test_kmeans.py`), and we optimized our final product via `tests/test_non_functional.py`, that automatically ran the prediction on all benchmark datasets.
- We implemented a profiler through `src/profiling.py` in order to estimate running times of individual running blocks of the code. It runs on a logic of time substraction between the start of the function call and the end of the function call. It gave us hints when certain algorithmical solutions became too slow for the context of the challenge. (i.e. staying within millisecond-level response rate)
