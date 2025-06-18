from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model.track(source=0, stream=True, show=False, save=True)  # predict on the webcam. the source can be a video file, online stream link or a webcam (0 for the default webcam)

for result in results:
    frame = result.orig_img.copy()
    
    if result.keypoints is None:
        continue

    kpts = result.keypoints.xy

    for i, person_kpts in enumerate(kpts):
        if len(person_kpts) < 2:
            continue
        
        # Use nose (keypoint 0) and neck (keypoint 1 or shoulders average) to define head direction
        try:
            nose = person_kpts[0]
            left_eye = person_kpts[1]
            right_eye = person_kpts[2]
            left_shoulder = person_kpts[5]
            right_shoulder = person_kpts[6]

            for x, y in person_kpts:
                x = int(x.item())
                y = int(y.item())
                cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            
            # Midpoint of shoulders as "torso direction"
            neck = (left_shoulder + right_shoulder) / 2

            print(f"Person {i}: Nose at {nose}, Neck at {neck}")

            # Direction vector
            direction = nose - neck
            direction[1] = direction[1] * 0.10  # Adjust vertical component to emphasize horizontal direction
            direction_unit = direction / (np.linalg.norm(direction) + 1e-6)

            print(f"Person {i}: Direction vector {direction}, Unit vector {direction_unit}")

            # # Draw head direction
            nose_np = nose.cpu().numpy()
            end_point = (nose_np + 50 * direction_unit.cpu().numpy()).astype(int)
            cv2.arrowedLine(frame, tuple(nose_np.astype(int)), tuple(end_point), (255, 0, 0), 2)


            # Label the person
            cv2.putText(frame, f'Person {i}', tuple(nose_np.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        except Exception as e:
            print(f"Error processing person {i}: {e}")
            continue

    cv2.imshow("Who is looking at whom?", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
