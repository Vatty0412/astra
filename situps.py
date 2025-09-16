import cv2
import mediapipe as mp
import numpy as np

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    """
    Calculates the angle in degrees between three points.
    
    Args:
        a (tuple): Coordinates of the first point (e.g., shoulder).
        b (tuple): Coordinates of the middle point (e.g., hip).
        c (tuple): Coordinates of the third point (e.g., knee).
        
    Returns:
        float: The calculated angle in degrees.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # Third point
    
    # Calculate the vectors
    ba = a - b
    bc = c - b
        
    # Calculate the cosine of the angle using the dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Handle floating point inaccuracies
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # Convert the angle to degrees
    angle = np.degrees(angle)
    
    return angle

# Function to calculate a score based on the hip angle
def calculate_score(min_hip_angle):
    """
    Calculates a sit-up score based on the minimum hip angle achieved.
    The score is a linear function that gives 100 for an angle <= 45 degrees
    and 0 for an angle >= 90 degrees.
    
    Args:
        min_hip_angle (float): The minimum angle at the hip during the sit-up.
    
    Returns:
        int: The calculated score (from 0 to 100).
    """
    perfect_angle = 45.0
    fail_angle = 90.0
    
    if min_hip_angle <= perfect_angle:
        return 100
    if min_hip_angle >= fail_angle:
        return 0
    
    # Linear interpolation for scores between perfect and fail angles
    # The score decreases as the angle increases
    score = 100 - ((min_hip_angle - perfect_angle) / (fail_angle - perfect_angle)) * 100
    return int(max(0, score))

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize video capture
# Replace 'your_video.mp4' with the path to your video file.
video_path = 'your_video.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Sit-up counter and scoring variables
counter = 0
total_score = 0
scores = []
stage = "down"  # Can be 'down' or 'up'
min_angle_per_rep = 180  # Initialize with a high value

print("Sit-up Counter is running. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Pose
    results = pose.process(frame_rgb)
    
    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # Extract landmarks for the key joints (left side)
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for hip, shoulder, and knee
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            # Calculate the angle at the hip
            angle = calculate_angle(shoulder, hip, knee)
            
            # Visualize the angle on the frame
            cv2.putText(frame, str(int(angle)), 
                        tuple(np.multiply(hip, [frame.shape[1], frame.shape[0]]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Sit-up counting and scoring logic
            # The angle will be large when the person is lying down (close to 180 degrees)
            # The angle will be small when the person is sitting up
            
            # Update min_angle_per_rep if we are in the 'up' stage
            if stage == "up":
                min_angle_per_rep = min(min_angle_per_rep, angle)

            # Check if the person is 'down' (angle > 160 degrees)
            if angle > 160:
                stage = "down"
            
            # Check if the person is 'up' and the stage was 'down'
            if angle < 90 and stage == 'down':
                stage = "up"
                counter += 1  # Increment the counter
                
                # Calculate score for the completed sit-up and reset min_angle_per_rep
                current_score = calculate_score(min_angle_per_rep)
                scores.append(current_score)
                total_score += current_score
                min_angle_per_rep = 180  # Reset for the next repetition

                print(f"Sit-up count: {counter}, Score: {current_score}")
        
        except Exception as e:
            # Handle cases where landmarks might be missing in a frame
            print(f"An error occurred: {e}")
            pass
            
    # Display sit-up count and current score on the screen
    cv2.putText(frame, f"Sit-ups: {counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the score for the last sit-up and the average score
    if scores:
        last_score = scores[-1]
        avg_score = total_score / len(scores)
        cv2.putText(frame, f"Last Score: {last_score}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Avg. Score: {avg_score:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Display the processed frame
    cv2.imshow('Sit-up Grader', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
