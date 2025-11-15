import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load video
cap = cv2.VideoCapture("fan_output.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

print(fps)

# Crop region for display (larger area to see context)
## This could be acquired from the object detection model
"""
### drone_idle
crop_x_min = 500
crop_x_max = 700
crop_y_min = 300
crop_y_max = 450

### fan 
crop_x_min = 550
crop_x_max = 700
crop_y_min = 250
crop_y_max = 400
"""

crop_x_min = 550
crop_x_max = 700
crop_y_min = 250
crop_y_max = 250

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
    roi_x_min = crop_x_min #+ (crop_width // 2)  # right half
    roi_x_max = crop_x_max
    roi_y_min = crop_y_min                      # full height
    roi_y_max = crop_y_max                     # bottom

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
    # Record intensity if recording (count black pixels)
    if recording:
        roi = gray[roi_y_min:roi_y_max, roi_x_min:roi_x_max]

        intensity_value = np.mean(roi)
        intensity.append(intensity_value)
    
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
if len(intensity) > 20:
    intensity = np.array(intensity, dtype=float)
    
    signal = intensity - np.mean(intensity)
    
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1 / fps)

    
    plt.plot(freqs, fft_vals)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT of Signal')
    plt.show()
    
    
    # Find strongest non-zero frequency
    peak_idx = np.argmax(fft_vals[1:]) + 1
    rot_freq_hz = freqs[peak_idx]

    # RPM calculation
    blade_count = 3   # set your blade count
    rpm = (rot_freq_hz * 60) / blade_count

    print(f"Estimated RPM: {rpm:.2f}")
    """
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
    """