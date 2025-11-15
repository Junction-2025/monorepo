import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("fan_output.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

print(fps)

# Crop region for display (larger area to see context)
crop_x_min = 550
crop_x_max = 700
crop_y_min = 250
crop_y_max = 400
crop_height = crop_y_max - crop_y_min
crop_width = crop_x_max - crop_x_min

intensity = []
frame_count = 0

print("Press 'q' to quit, 's' to start recording intensity")
recording = False

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame_count += 1
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Crop the frame for display
    cropped = gray[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    
    # Define rectangular ROI 5
    roi_x_min = crop_x_min + (crop_width // 2)  # center of cropped width
    roi_x_max = crop_x_max                       # right edge
    roi_y_min = crop_y_min + (crop_height // 2)                    # top
    roi_y_max = crop_y_max                       # bottom

    # Draw rectangle on cropped frame for visualization
    roi_x_in_crop = roi_x_min - crop_x_min
    roi_y_in_crop = roi_y_min - crop_y_min
    cv2.rectangle(cropped,
                (roi_x_in_crop, roi_y_in_crop),
                (roi_x_in_crop + (roi_x_max - roi_x_min),
                roi_y_in_crop + (roi_y_max - roi_y_min)),
                (255, 255, 255), 1)

    cv2.imshow("Cropped View", cropped)
    
    # Record intensity if recording (average over entire ROI)
    if recording:
        roi = gray[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        intensity.append(np.mean(roi))
    
    # Keyboard controls
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        recording = True
        print("Started recording intensity...")
    elif key == ord('p'):
        recording = False
        print("Paused recording")

cap.release()
cv2.destroyAllWindows()

# Process intensity data if recorded
if len(intensity) > 0:
    intensity = np.array(intensity)
    
    # Optional: threshold for binary signal
    binary = (intensity > 128).astype(int)
    
    # Detect rising edges
    edges = np.diff(binary) > 0
    peak_frames = np.where(edges)[0]
    
    # Calculate average period in frames
    if len(peak_frames) > 1:
        periods = np.diff(peak_frames)
        avg_period_frames = np.mean(periods)
        f_hz = fps / avg_period_frames
        blade_count = 3  # set your propeller blade count
        rpm = f_hz * 60 / blade_count
        print(f"Estimated RPM: {rpm:.2f}")
    else:
        print("Not enough peaks detected to estimate RPM")
else:
    print("No intensity data recorded")