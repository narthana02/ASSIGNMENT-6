#!/usr/bin/env python
# coding: utf-8

# # TASK - 5

# # Steps to Calculate Dwell Time for Each Person in the Entire Video
# 
# Load the video and initialize the background subtractor.
# 
# Detect and track objects in each frame.
# 
# Assign unique IDs to each detected object.
# 
# Track the time each object is visible in the frame.
# 
# Output the total dwell time for each person detected in the video.

# In[10]:


import cv2
import numpy as np
from collections import defaultdict

# Path to your video file
video_path = "C:\\Users\\Narthana\\Downloads\\278ce34f-b588-43fc-a2ca-12083b012498 (1).mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=50, detectShadows=False)

# Frames per second of the video to calculate time per frame
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps  # Time per frame in seconds

# Dictionary to store dwell time for each unique object
dwell_times = defaultdict(float)
object_centers = {}  # Dictionary to store the last known position of each object

object_id_count = 0  # Counter to assign unique IDs to each object

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Additional filtering to clean up the foreground mask
    fgmask = cv2.medianBlur(fgmask, 5)  # Reduce noise with median blur

    # Threshold to create binary image, fine-tune based on your scene
    _, thresh = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_frame_centers = []  # Track the current frame's object centers

    for cnt in contours:
        # Filter out small contours
        if cv2.contourArea(cnt) < 1000:  # Adjust contour area threshold if needed
            continue

        # Get bounding box for each object
        x, y, w, h = cv2.boundingRect(cnt)
        center = (x + w // 2, y + h // 2)

        # Check if the object matches any previous object based on proximity
        matched_object_id = None
        for obj_id, prev_center in object_centers.items():
            if np.linalg.norm(np.array(center) - np.array(prev_center)) < 50:  # Adjust distance threshold if needed
                matched_object_id = obj_id
                break

        # If no match found, assign a new ID
        if matched_object_id is None:
            matched_object_id = object_id_count
            object_id_count += 1

        # Update the center position of the matched object
        object_centers[matched_object_id] = center
        current_frame_centers.append(matched_object_id)

        # Increment the dwell time for this object by the time per frame
        dwell_times[matched_object_id] += frame_time

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {matched_object_id} Time: {dwell_times[matched_object_id]:.2f}s", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Remove objects that are no longer detected in the frame
    object_centers = {obj_id: center for obj_id, center in object_centers.items() if obj_id in current_frame_centers}

    # Display the frame with detections
    cv2.imshow("Dwell Time Detection", frame)

    # Exit condition
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Print the total dwell time for each object/person detected in the video
print("Total Dwell Time for Each Detected Object:")
for obj_id, time in dwell_times.items():
    print(f"Object ID {obj_id}: {time:.2f} seconds")


# # TASK - 2

# 
# Video Input:
# 
# The video is loaded using OpenCV (cv2.VideoCapture), and the path to the video file is specified (video_path).
# Background Subtraction:
# 
# A background subtractor (cv2.createBackgroundSubtractorMOG2) is used to detect moving objects by comparing the current frame to the background.
# Frame Rate and Interval:
# 
# The frame rate of the video is captured using cap.get(cv2.CAP_PROP_FPS). For each 1-minute interval, the number of frames is calculated (interval_frames), assuming the video runs at 30 FPS.
# People Detection:
# 
# Each frame is processed by the detect_people() function, which:
# Applies background subtraction to identify moving objects.
# Converts the foreground mask into a binary image using cv2.threshold.
# Finds contours and filters out small contours (noise).
# For each valid contour, it counts people and draws bounding boxes around them.
# The number of detected people (people_count) is stored for each frame.
# Tracking People Over Time:
# 
# A deque (frame_queue) stores the people counts for each frame in the current interval (1-minute window).
# Once the queue is full (after processing frames for one minute), the sum of the counts is recorded in people_counts, and the corresponding time interval in time_intervals.
# Peak Detection:
# 
# The peak shopping duration is identified by finding the time interval with the maximum number of detected people. This is done by comparing each total count and storing the corresponding interval (peak_time_interval).
# Frames that contribute to this peak interval are stored in peak_frames and displayed later.
# Frame Display:
# 
# Each processed frame (with bounding boxes) is displayed using cv2.imshow.
# Exit Condition:
# 
# The loop continues processing until the end of the video or the user presses the 'q' key to exit.
# Peak Duration Output:
# 
# After processing the video, the peak shopping duration (in seconds) and the number of people detected in the peak frames are printed.

# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Load the video
video_path = "C:\\Users\\Narthana\\Downloads\\278ce34f-b588-43fc-a2ca-12083b012498 (1).mp4"
cap = cv2.VideoCapture(video_path)

# Background subtraction method (optional: use other methods like optical flow)
fgbg = cv2.createBackgroundSubtractorMOG2()

# Set frame rate for 1-minute intervals (assuming 30 FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
interval_frames = int(fps * 60)  # 60 seconds worth of frames (1-minute interval)

# To store counts over time
people_counts = []
time_intervals = []

# To store frames for peak duration
peak_frames = []

# Initialize a deque to store frames for the current interval
frame_queue = deque(maxlen=interval_frames)

# Function to detect people in a frame (simplified using background subtraction)
def detect_people(frame):
    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    people_count = 0
    bounding_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filter by minimum area to avoid noise
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
            people_count += 1
    return people_count, bounding_boxes

# Loop through the video frames
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for motion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people in the current frame
    people_count, bounding_boxes = detect_people(gray_frame)
    
    # Draw bounding boxes on the frame
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
    
    # Display the frame with bounding boxes
    cv2.imshow("Detected People", frame)

    # Add to current interval's frame queue
    frame_queue.append(people_count)

    # When the interval is full (1 minute), calculate total people count
    if len(frame_queue) == interval_frames:
        total_people = sum(frame_queue)
        people_counts.append(total_people)
        time_intervals.append(frame_idx / fps)  # Time in seconds for current interval
        
        # Update peak frames
        if total_people == max(people_counts):
            peak_frames = list(frame_queue)
    
    frame_idx += 1

    # Break on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Display peak duration information
peak_time_interval = time_intervals[people_counts.index(max(people_counts))]
print(f"Peak Shopping Duration: {peak_time_interval} seconds")

# Display some of the frames from the peak interval
for idx, frame_count in enumerate(peak_frames):
    print(f"Peak Frame {idx+1}: {frame_count} people detected")


# # TASK - 1

# This code tracks a person in a video using background subtraction and contour detection.
# 
# Video Loading: The video is loaded using OpenCV's cv2.VideoCapture.
# 
# Background Subtraction: A background subtractor (cv2.createBackgroundSubtractorMOG2) is applied to detect moving objects in the frame by subtracting the static background.
# 
# Noise Reduction: Morphological operations (cv2.MORPH_OPEN and cv2.dilate) are applied to remove noise and smooth the foreground mask.
# 
# Contour Detection: The cv2.findContours function finds contours (regions of connected pixels) in the foreground mask.
# 
# Person Detection: Each contour is evaluated based on its area and aspect ratio (height/width). The largest contour that matches the shape and size criteria is assumed to be the person.
# 
# Tracking: The center of the detected person's bounding box is saved and used to draw a tracking path over the frames.
# 
# Visualization: The bounding box around the person is drawn, along with the tracking path, if applicable. The processed frame is displayed using cv2.imshow.

# In[32]:


import cv2
import numpy as np

# Load the video file
video_path = "C:\\Users\\Narthana\\Downloads\\278ce34f-b588-43fc-a2ca-12083b012498 (1).mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=True)

# Variables for tracking
tracking_points = []
MAX_TRACK_HISTORY = 10

# Parameters for filtering
MIN_CONTOUR_AREA = 1500  # Minimum area to consider (adjust as needed)
MIN_ASPECT_RATIO = 0.4   # Minimum aspect ratio (height/width) for a person
MAX_ASPECT_RATIO = 2.5   # Maximum aspect ratio for a person

def process_frame(frame):
    # Step 1: Apply background subtraction to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)
    
    # Step 2: Remove noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    
    # Step 3: Find contours in the mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 4: Filter contours to detect the person based on area and aspect ratio
    person_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / float(w)

            # Check if contour meets size and shape criteria for a person
            if MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
                if area > max_area:
                    max_area = area
                    person_contour = contour

    # Step 5: Draw bounding box around the largest valid contour if found
    if person_contour is not None:
        x, y, w, h = cv2.boundingRect(person_contour)
        center = (x + w // 2, y + h // 2)
        
        # Save center point for smooth tracking
        tracking_points.append(center)
        if len(tracking_points) > MAX_TRACK_HISTORY:
            tracking_points.pop(0)
        
        # Draw bounding box around detected person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    # Optional: Draw tracking path to show recent movement
    for i in range(1, len(tracking_points)):
        if tracking_points[i - 1] is None or tracking_points[i] is None:
            continue
        thickness = int(np.sqrt(MAX_TRACK_HISTORY / float(i + 1)) * 2.5)
        cv2.line(frame, tracking_points[i - 1], tracking_points[i], (0, 0, 255), thickness)

    return frame

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the current frame
    processed_frame = process_frame(frame)

    # Display the processed frame
    cv2.imshow("Person Tracking", processed_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()


# # TASK - 6

# This code performs branded car detection in a video by applying background subtraction and color-based filtering.
# 
# Video Loading: The video is loaded using cv2.VideoCapture(). If it fails to load, an error is raised.
# Background Subtraction: A background subtractor (cv2.createBackgroundSubtractorMOG2()) is used to detect moving objects (cars). A binary mask is created to highlight moving areas.
# Color Ranges Definition: Specific color ranges for black, white, brown, and red cars are defined in the HSV color space to filter cars based on their color.
# Car Detection Loop: Each frame is processed:
# Background Masking: The moving areas are detected using the background subtractor and thresholded to filter noise.
# Contour Detection: Contours of detected areas are found to identify possible cars.
# Bounding Box Creation: For each detected contour, a bounding box is drawn around the car.
# Color Detection: The region of interest (ROI) around the car is checked for the defined colors using color masking. If a significant portion of the car matches a color, it is labeled.
# Car Count Update: When a branded car is detected, a counter (car_count) is incremented.
# Display Frame: The processed frame with bounding boxes and color labels is displayed.
# Exit Condition: The loop exits if the user presses 'q'.
# Cleanup: The video capture object is released, and all OpenCV windows are closed.
# Final Output: The total count of detected branded cars is printed.

# In[4]:


import cv2
import numpy as np

# Path to the uploaded video
video_path = "C:\\Users\\Narthana\\Downloads\\youmarker.mp4"

# Load the video
cap = cv2.VideoCapture(video_path)

# Check if video loaded successfully
if not cap.isOpened():
    raise Exception("Error: Could not open video file.")

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()

# Define refined color ranges for black, white, brown, and red cars in HSV color space
color_ranges = {
    "black": ((0, 0, 0), (180, 255, 40)),  # Restricting upper limit for black
    "white": ((0, 0, 200), (180, 30, 255)),  # Narrowing white's brightness threshold
    "brown": ((10, 100, 50), (20, 255, 200)),  # More precise for brown range
    "red1": ((0, 70, 50), (10, 255, 255)),  # First range for red
    "red2": ((160, 70, 50), (180, 255, 255)),  # Second range for red
}

# Initialize car count
car_count = 0

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction to detect moving objects
    fgmask = fgbg.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours of the detected areas
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and process each contour
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:  # Minimum area to filter noise
            continue
        
        # Get bounding box for each detected car
        x, y, w, h = cv2.boundingRect(cnt)
        car_roi = frame[y:y+h, x:x+w]  # Region of interest (ROI) for the car
        
        # Convert ROI to HSV color space
        hsv_roi = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)
        
        # Check for branded car color match and set label
        color_label = None
        for color, (lower, upper) in color_ranges.items():
            if color == "red2":
                # Second range for red color (as red wraps in HSV)
                mask = cv2.inRange(hsv_roi, color_ranges["red1"][0], color_ranges["red1"][1]) | \
                       cv2.inRange(hsv_roi, color_ranges["red2"][0], color_ranges["red2"][1])
            else:
                mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
            
            # Check if a sufficient area within the ROI matches the color
            if cv2.countNonZero(mask) > (0.2 * w * h):  # At least 20% of the area matches
                color_label = color
                car_count += 1
                break

        # Draw bounding box and color label if a branded car was detected
        if color_label:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, color_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame with detections
    cv2.imshow('Branded Car Detection', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print the total count of detected branded cars
print(f"Total branded car detections: {car_count}")


# # TASK - 4

# In[8]:


import cv2

# Load the video
video_path = "C:\\Users\\Narthana\\Downloads\\278ce34f-b588-43fc-a2ca-12083b012498 (1).mp4"
cap = cv2.VideoCapture(video_path)

# Define region of interest (ROI) near the entrance
roi_x, roi_y, roi_width, roi_height = 100, 200, 200, 150  # Adjust based on entrance position

# Background subtraction for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

# Initialize counters for people entering and exiting
enter_count = 0
exit_count = 0

# Variables to hold movement direction
last_direction = None
direction_threshold = 30  # Minimum movement threshold for counting

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Define the ROI for detecting motion at the entrance
    roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Detect motion using background subtraction
    fg_mask = fgbg.apply(blurred_roi)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Find contours to identify moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Ignore small contours to avoid noise
        if cv2.contourArea(contour) < 500:
            continue

        # Draw bounding box around detected motion in the ROI
        x, y, w, h = cv2.boundingRect(contour)

        # Adjust the position of the bounding box vertically down
        roi_y += 10  # Increase the ROI's Y-coordinate to move down
        roi_height += 10  # Increase the height of the ROI

        # Ensure the ROI does not exceed the frame height
        roi_height = min(roi_height, frame.shape[0] - roi_y)

        # Draw the adjusted bounding box on the original frame
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (0, 255, 0), 2)

        # Calculate movement direction (up for entering, down for exiting)
        if last_direction is None:
            last_direction = y
        else:
            direction = y - last_direction
            if abs(direction) > direction_threshold:
                if direction < 0:
                    enter_count += 1
                    print(f"Person entered, Total Entered: {enter_count}")
                elif direction > 0:
                    exit_count += 1
                    print(f"Person exited, Total Exited: {exit_count}")
                last_direction = y

    # Display the frame with updated ROI and motion highlighted
    cv2.putText(frame, f"Entered: {enter_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Exited: {exit_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Shop Entrance", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Final count - Entered: {enter_count}, Exited: {exit_count}")


# In[4]:


import cv2
import numpy as np

# Load the reference image
reference_image = cv2.imread("C:\\Users\\Narthana\\Downloads\\Screenshot 2024-11-14 122059.png")

# Load the video
video_capture = cv2.VideoCapture("C:\\Users\\Narthana\\Downloads\\Elevator - Racism. It Stops With Me.mp4")


# In[5]:


# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect face in the reference image
gray_reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
faces_in_reference = face_cascade.detectMultiScale(gray_reference_image, scaleFactor=1.1, minNeighbors=5)

# If a face is detected, extract it
for (x, y, w, h) in faces_in_reference:
    reference_face = reference_image[y:y+h, x:x+w]


# In[6]:


# Load the Haar Cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Detect eyes in the reference image
eyes_in_reference = eye_cascade.detectMultiScale(gray_reference_image, scaleFactor=1.1, minNeighbors=5)
eye_features_reference = []

for (ex, ey, ew, eh) in eyes_in_reference:
    eye_features_reference.append((ex, ey, ew, eh))


# In[7]:


while True:
    # Read each frame of the video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the video frame
    faces_in_video = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_in_video:
        # Extract face region from the video frame
        face_in_video = frame[y:y+h, x:x+w]

        # Detect eyes in the video face region
        eyes_in_video = eye_cascade.detectMultiScale(gray_frame[y:y+h, x:x+w], scaleFactor=1.1, minNeighbors=5)
        eye_features_video = []

        for (ex, ey, ew, eh) in eyes_in_video:
            eye_features_video.append((ex, ey, ew, eh))

        # Compare the features (simple distance between eye coordinates)
        if len(eye_features_reference) == len(eye_features_video):
            match = True
            for ref_eye, video_eye in zip(eye_features_reference, eye_features_video):
                distance = np.linalg.norm(np.array(ref_eye[:2]) - np.array(video_eye[:2]))
                if distance > 50:  # Set a threshold for feature matching
                    match = False
                    break

            # If a match is found, highlight the face in the video
            if match:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Match Found", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the video with highlighted faces
    cv2.imshow('Video', frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()


# # TASK - 3

# In[9]:


import cv2
import numpy as np
import os


# Set paths for reference image and video
reference_image_path = "C:\\Users\\Narthana\\Downloads\\Screenshot 2024-11-14 122059.png"
video_path = "C:\\Users\\Narthana\\Downloads\\Elevator - Racism. It Stops With Me.mp4"

# Specify output folder for matched frames
output_folder = "C:\\Users\\Narthana\\Downloads\\output"
os.makedirs(output_folder, exist_ok=True)

# Load the reference image and convert it to grayscale
reference_image = cv2.imread(reference_image_path)
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect face in the reference image
ref_faces = face_cascade.detectMultiScale(reference_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Ensure there's at least one face detected in the reference image
if len(ref_faces) == 0:
    print("No faces found in the reference image.")
    exit()
else:
    print(f"{len(ref_faces)} face(s) detected in the reference image.")

# Extract the detected face from the reference image (use the largest face)
x, y, w, h = max(ref_faces, key=lambda face: face[2] * face[3])
reference_face = reference_gray[y:y+h, x:x+w]

# Load the video
cap = cv2.VideoCapture(video_path)

# Initialize a frame counter
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    match_found = False  # Flag to check if any match is found in the current frame

    # Process each detected face in the frame
    for (fx, fy, fw, fh) in faces:
        # Extract the face from the frame
        face_in_frame = gray_frame[fy:fy+fh, fx:fx+fw]

        # Resize the reference face and detected face to the same size for comparison
        resized_reference = cv2.resize(reference_face, (fw, fh))
        match_result = cv2.matchTemplate(face_in_frame, resized_reference, cv2.TM_CCOEFF_NORMED)
        _, match_val, _, _ = cv2.minMaxLoc(match_result)

        # Check if match value exceeds threshold (indicates a match)
        match_threshold = 0.7
        if match_val > match_threshold:
            # Draw a rectangle around the matching face in the frame
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            cv2.putText(frame, f'Match: {match_val:.2f}', (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Match found in frame {frame_count} with similarity score: {match_val:.2f}")

            # Save the frame with match to the output folder
            output_frame_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(output_frame_path, frame)
            match_found = True

    # Display only frames with a detected match
    if match_found:
        cv2.imshow('Matching Frame', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




