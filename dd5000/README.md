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


## RPM Calculation

- Events are grouped into 1ms time windows
- "On" events (positive polarity) within the ROI are counted per window using boolean masking
- Creates a time-series signal of event activity (100 samples collected)
- FFT analysis (`np.fft.rfft()`) identifies the dominant frequency peak (blade passage rate)
- RPM = `(frequency_hz * 60) / blade_count` (converts blade passages/min to rotations/min)



## YOLO model finetuning for event-camera usecase


YOLO models are the best standards known for object detection models. However, using them out of the box for event cameras, doesn't make sense, as we want to process it fast.

Hence, what we did was, fine-tune the models for event data for drones, and as we shown below, achieved fast inference speeds to be able to use them for the usecase.

Also, we have shown (in the yt video) use of BoT-SORT for tracking that can be done as well.



This table summarises the performance of the trained models using three key metrics:
- **mAP50** (at final epoch)
- **Inference speed** (on both GPU and Apple M4 Max)


## Model Comparison

these are ran on validation sets, and inference time taken on those


| Model File                     | mAP50 (Final Epoch) | Inference Speed (GPU) | Inference Speed (M4 Max) | Notes |
|-------------------------------|----------------------|-------------------------|----------------------------|--------|
| **best-final.pt**             | **0.95138**          | **6.1 ms**             | **—**                      | Best fine-tuned model. Combines different datasets |
| **best-custom.pt**            | **0.99500**          | **7.3 ms**             | **—**                      | Trained on custom dataset extracted from .dat file from FRED dataset |
| **best-m.pt**                 | **0.92544**          | **7.6 ms**             | **—**                      | Same model but finetuned on yolo11(m).pt |
| **best-custom-added-drone.pt**| **0.99500**          | **7.4 ms**             | **—**                      | Added extra drone examples; similar accuracy to best-custom.pt |





## Technical achievements
- We accurately detect RPM on moving drones
    - On `drone_moving.dat` we capture average RPM in the scale of `5680.43`, `6192.86`, though we also detected `5425.53`, which is slightly outside of the range
    - On `drone_idle.dat` we capture average RPM in the scale of `5631.25`, `5608.16`, `5632.65`. The scale is a lot tighter than the moving case. 
- In our experiments, the fine-tuned YOLO model could detect drones frame-by-frame, running on a normal Macbook, in only 8ms. This is a significant finding, as even the Sensofusion challenge providers did not believe this could be achieved.
    - With a YOLO-based drone detection system, it's possible to label enemy drones accurately in real time. It is possible to teach YOLO models to detect different drone types.
- We completely implemented the KNN propeller detection concept from [this publication](https://arxiv.org/pdf/2209.02205) and integrated it with RPM estimation via a stateful blade count tracker (`src/blade_tracker.py`).

## Development philosophy
- We utilized a mild functional development philosophy. The aim was to create composable modules that could be called on their lonesome using `uv -m`. (`src/main.py`, `src/kmeans.py`, `src/yolo.py`)
- We did some software testing where it made sense in the interest of time. The k-means implementation needed testing to verify that it works (`tests/test_kmeans.py`), and we optimized our final product via `tests/test_non_functional.py`, that automatically ran the prediction on all benchmark datasets.
![alt text](docs/image.png)
- We implemented a profiler through `src/profiling.py` in order to estimate running times of individual running blocks of the code. It runs on the logic of time substraction between the start of the function call and the end of the function call. It gave us hints when certain algorithmical solutions became too slow for the context of the challenge, helping us to remove expensive solutions early. (i.e. staying within millisecond-level response rate)
```
=== Main function benchmarks ===
INFO - extract_roi_intensity: avg=0.01 ms, min=0.01 ms, max=0.23 ms, n=2071
INFO - identification_delay: avg=95.41 ms, min=41.12 ms, max=1369.72 ms, n=207
INFO - estimate_rpm_from_signal: avg=0.04 ms, min=0.02 ms, max=0.72 ms, n=69
```
You can see that we're able to run this feature-rich inference in 100 milliseconds. The code can be optimized a lot further by our estimates as well.