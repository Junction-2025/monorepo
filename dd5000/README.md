# Drone Detector 5000 (Challenge 1)

Drone Detector 5000 is a resilient, fast,  and efficient way of detecting drones.  It works in dark and high-contrast lighting conditions,  detects drones in a matter of milliseconds, and provides detailed information on the number of rotors, blades, and the RPM, thus enabling the narrowing down of the drone type.

The project is based on a simple event loop (`src/main.py`):

- Events are taken into the system whenever they're created (`src/evio_lib/`)
- At a configurable interval, temporally concatenate the events into a frame (`src/main.py`)
- The frame is fed to a fine-tuned YOLO model (`src/yolo.py`)
    - YOLO detects if the event contains a drone (`src/yolo.py`)
        - If not, discard frame
        - If yes, continue
- YOLO gives us a bounding box containing the drone (`src/yolo.py`)
- We use KMeans to detect clusters of high frequency events inside of the bounding box. These will be the drone's propellers. (`src/kmeans.py`)
    - KMeans is customized to be intialized via specific centroids (`src/kmeans.py`)
    - KMeans auto-adjusts its' chosen `k` (`src/kmeans.py`)
- Once we have `k` clusters of high event frequency, we process each of these clusters separately (`src/main.py`)
- We once again run KMeans to detect blades of the rotor, with similar principles. (`src/kmeans.py`)
- We use the blade count for each rotor to compute RPM (`src/rpm.py`)
    - Blade counts from KNN detection are tracked over time using a stateful tracker (`src/blade_tracker.py`)
    - The tracker uses statistical methods (mode/median) to provide a robust estimate
    - RPM calculations dynamically use the tracked blade count instead of hardcoded values

## Technical achievements
- We accurately detect RPM on moving drones
    - On `drone_moving.dat` we capture average RPM in the scale of `5680.43`, `6192.86`, though we also detected `5425.53`, which is slightly outside of the range
    - On `drone_idle.dat` we capture average RPM in the scale of `5631.25`, `5608.16`, `5632.65`. The scale is a lot tighter than the moving case. 
- In our experiments, the fine-tuned YOLO model could detect drones frame-by-frame, running on a normal Macbook, in only 8ms. This is a significant finding, as even the Sensofusion challenge providers did not believe this could be achieved.
    - With a YOLO-based drone detection system, it's possible to label enemy drones accurately in real time. It is possible to teach YOLO models to detect different drone types.
- We completely implemented the KNN propeller detection concept from [this publication](https://arxiv.org/pdf/2209.02205) and almost integrated it with RPM estimation via a stateful blade count tracker (`src/blade_tracker.py`).

## Development philosophy
- We utilized a mild functional development philosophy. The aim was to create composable modules that could be called on their lonesome using `uv -m`. (`src/main.py`, `src/kmeans.py`, `src/yolo.py`)
- We did some software testing where it made sense in the interest of time. The k-means implementation needed testing to verify that it works (`tests/test_kmeans.py`), and we optimized our final product via `tests/test_non_functional.py`, that automatically ran the prediction on all benchmark datasets; the nft's do not work for the final state of things. 
- We implemented a profiler through `src/profiling.py` in order to estimate running times of individual running blocks of the code. It runs on the logic of time substraction between the start of the function call and the end of the function call. It gave us hints when certain algorithmical solutions became too slow for the context of the challenge, helping us to remove expensive solutions early. (i.e. staying within millisecond-level response rate)
