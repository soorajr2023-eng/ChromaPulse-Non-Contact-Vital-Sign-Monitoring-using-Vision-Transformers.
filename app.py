import cv2
import mediapipe as mp
import numpy as np

def main():
    # Initialize MediaPipe for high-speed face tracking
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    cap = cv2.VideoCapture(0)
    print("ChromaPulse Active. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Image Pre-processing for Computer Vision
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Landmark 10 is the center of the forehead
                h, w, _ = frame.shape
                cx = int(face_landmarks.landmark[10].x * w)
                cy = int(face_landmarks.landmark[10].y * h)

                # Draw the tracking 'sensor'
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(frame, "Analyzing Forehead ROI", (cx + 20, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('ChromaPulse - VIT Computer Vision BYOP', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
