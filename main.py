import cv2
import mediapipe as mp

# Accessing the first webcam with index 0
cam = cv2.VideoCapture(0)

# Face Detection
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# running infinite loop for camera for capturing video
while True:
    _, frame = cam.read()  # Reading output from camera
    frame = cv2.flip(frame, 1)  # flipping the frame vertically
    rgb_frame = cv2.cvtColor(frame,
                             cv2.COLOR_BGR2RGB)  # Converting video to grey becoz It is easy to trace face in grey
    output = face_mesh.process(rgb_frame)
    landmarksPoints = output.multi_face_landmarks  # Detecting landmarks on face like nose, forehead etc
    frameHeight, frameWidth, _ = frame.shape  # Getting dimensions of frame
    # print(landmarksPoints)  # can remove this line after testing
    if landmarksPoints:
        landmarks = landmarksPoints[0].landmark  # Detecting only one face in video
        for landmark in landmarks[474:478]:
            x = int(landmark.x * frameWidth)
            y = int(landmark.y * frameHeight)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))  # Drawing circles on face; this accepting 3 args : a) frame b) circle coordinates c) radius d) color
            # print(x, y)

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frameWidth)
            y = int(landmark.y * frameHeight)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        if(left[0].y - left[1].y) < 0.004:
            print('You Are Sleeping !!!')
    cv2.imshow('Detecting Eyes', frame)  # Detecting image from video { im = image }
    cv2.waitKey(1)