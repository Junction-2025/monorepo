# Drone Detector 5000

Drone Detector 5000 is based on a simple event loop.

- Events are taken into the system whenever they're created
- At a configurable interval, temporally concatenate the events into a frame
- The frame is fed to a fine-tuned YOLO algorithm
    - YOLO detects if the event contains a drone
        - If not, discard frame
        - If yet, continue
- YOLO gives us a bounding box containing the drone
- We use KMeans to detect clusters of high frequency events inside of the bounding box. These will be the drone's propellers.
    - KMeans is customized to be intialized via customized centroids
    - KMeans auto-adjusts its' chosen `k`
- Once we have `k` clusters of high event frequency, we process each of these clusters separately
- We once again run KMeans to detect blades of the rotor, with similar principles.
- We use the blade count for each toror to compute RPM using the FFT
