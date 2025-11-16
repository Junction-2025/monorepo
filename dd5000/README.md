# Drone Detector 5000 (Challenge 1)

Drone Detector 5000 is based on a simple event loop.

- Events are taken into the system whenever they're created
- At a configurable interval, temporally concatenate the events into a frame
- The frame is fed to a fine-tuned YOLO model
    - YOLO detects if the event contains a drone
        - If not, discard frame
        - If yet, continue
- YOLO gives us a bounding box containing the drone
- We use KMeans to detect clusters of high frequency events inside of the bounding box. These will be the drone's propellers.
    - KMeans is customized to be intialized via customized centroids
    - KMeans auto-adjusts its' chosen `k`
- Once we have `k` clusters of high event frequency, we process each of these clusters separately
- We once again run KMeans to detect blades of the rotor, with similar principles.
- We use the blade count for each toror to compute RPM

## Outcomes

## Technical Specifications
In our experiments, the fine-tuned YOLO model could detect drones frame-by-frame, running on a normal Macbook, in only 8ms. This is a significant finding, as even the Sensofusion challenge providers did not believe this could be achieved. 

## Development philosophy
- We utilized a mild functional development philosophy. The aim was to create composable modules that could be called on their lonesome using `uv -m`.
- We did some software testing where it made sense in the interest of time. The k-means implementation needed testing to verify that it works, and we optimized our final product via `tests/test_nonfunctional.py`, that automatically ran the prediction on all benchmark datasets. 
- We implemented a profiler
