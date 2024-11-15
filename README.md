# ASSIGNMENT-6

Task 1: Tagging a Person in Videos Taken in a Situation
Objective:
Tag and track a person across frames in a video based on appearance using traditional image processing techniques.

Task Description:

Load Video:

Load the provided video file using OpenCV.
Person Detection:

Use background subtraction or frame differencing to detect moving objects (people) in the video.
Apply basic feature extraction (color histograms, edge features) to isolate the target person based on appearance.
Person Tracking:

Implement a tracking algorithm like centroid-based tracking or optical flow to tag and track the detected person across the video frames.
Tagging Output:

Visualize the video with a bounding box and label/tag around the identified person as they move through the frames.
Task 2: Strategic Marketing – Peak Shopping Duration
Objective:
Analyze a video of a shopping area to identify the peak duration when the most people are shopping.

Task Description:

Load Video:

Load the surveillance video from a shopping area.
People Detection:

Use frame differencing or optical flow to detect motion and identify people entering the frame.
Count the number of people in each frame based on detected regions.
Peak Duration Identification:

Calculate the total number of people in the shopping area for each time period (e.g., 10-minute intervals).
Plot the number of people over time and identify the time interval with the highest count of people.
Result:

Provide a summary of the peak shopping duration and display the corresponding frames from the video.
Task 3: Facial Recognition to Check Fraud Cases
Objective:
Identify a suspect by comparing their facial features to a reference image to check for fraud cases, using traditional facial recognition techniques.

Task Description:

Load Images and Video:

Load the reference image of the suspect and a video showing multiple faces.
Face Detection:

Use Haar Cascades to detect faces in both the reference image and the video.
Feature Matching:

Extract facial features using edge detection or geometric facial features (eye spacing, nose length, etc.).
Compare the features of the faces in the video with the reference face to check for a match.
Result:

Output the frames where a match is found and highlight the detected face in the video.
Task 4: Number of People Entering and Exiting the Shop
Objective:
Count the number of people entering and exiting a shop based on video footage, using basic motion detection techniques.

Task Description:

Load Video:

Load the provided surveillance video of the shop entrance.
Motion Detection:

Use frame differencing or optical flow to detect motion as people enter and exit the shop.
Define a region of interest (ROI) near the entrance to focus on counting people.
Counting People:

Track the direction of motion (inward or outward) based on detected motion in the ROI.
Increment a counter for each person entering and exiting.
Result:

Display the total number of people who entered and exited the shop during the recorded period.
Task 5: Dwelling Time in a Shopping Mall
Objective:
Measure the amount of time a person or object dwells in a certain area of the shopping mall using video footage.

Task Description:

Load Video:

Load the surveillance video of the shopping mall.
Object/Person Detection:

Use background subtraction or motion detection to detect and track objects or people in the video.
Dwelling Time Calculation:

Set a region of interest (ROI) in the video representing a specific area of the mall.
Track the time each detected person/object spends in the ROI.
Result:

Display the total dwelling time for each person/object detected in the ROI.
Task 6: Spotting and Counting a Branded Car in a Video
Objective:
Identify and count the number of branded cars (e.g., a specific logo or color) in a video sequence using feature-based matching.

Task Description:

Load Video:

Load the provided video showing vehicles.
Car Detection:

Use background subtraction or motion detection to detect moving cars in the video.
Feature Matching:

Use color-based detection or template matching to identify the specific branded car (e.g., a car with a specific color or logo).
Track the occurrence of this branded car across the video frames.
Counting:

Count the number of times the branded car appears in the video.
Result:

Output the total count and display the frames where the branded car is detected.
