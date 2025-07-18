from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model.track(source=0, stream=True, show=True, save=True)  # predict on the webcam. the source can be a video file, online stream link or a webcam (0 for the default webcam)

SKELETON = {
    0 : [1, 2],
    1 : [3],
    2 : [4],
    5 : [6, 7, 11],
    6 : [8, 12],
    7 : [9],
    8 : [10],
    11 : [12, 13],
    12 : [14],
    13 : [15],
    14 : [16]
}

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MJPG', 'MP4V', etc.
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # adjust size/FPS

for result in results:
    frame = result.orig_img.copy()
    
    if result.keypoints is None:
        continue

    boxes = result.boxes  # bounding boxes for each detection

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

            conf_threshold = 0.5  # adjust this as needed (0.5 = 50% confidence)

            for x, y, c in zip(person_kpts[:, 0], person_kpts[:, 1], result.keypoints.conf[i]):
                if c.item() > conf_threshold:  # only draw if confidence > threshold
                    cv2.circle(frame, (int(x.item()), int(y.item())), radius=5, color=(0, 255, 0), thickness=-1)

            x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()  # convert tensor to list
            conf = float(result.boxes.conf[i])
            cls_id = int(result.boxes.cls[i])  # should be 0 for 'person'

            # draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            # label
            label = f"Person {i} ({conf:.2f})"
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
            
            confidences = result.keypoints.conf[i]

            for start, ends in SKELETON.items():
                for end in ends:
                    if confidences[start].item() > conf_threshold and confidences[end].item() > conf_threshold:
                        x1 = person_kpts[start][0].item()
                        y1 = person_kpts[start][1].item()
                        x2 = person_kpts[end][0].item()
                        y2 = person_kpts[end][1].item()

                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


            # for x, y in person_kpts:
            #     x = int(x.item())
            #     y = int(y.item())
            #     cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
# 
            # for start, ends in SKELETON.items():
            #     for end in ends:
            #         x1 = person_kpts[start][0].item()
            #         y1 = person_kpts[start][1].item()
            #         x2 = person_kpts[end][0].item()
            #         y2 = person_kpts[end][1].item()

            #         cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            
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


            # # Label the person
            # cv2.putText(frame, f'Person {i}', tuple(nose_np.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        except Exception as e:
            print(f"Error processing person {i}: {e}")
            continue

    out.write(frame)
    cv2.imshow("Who is looking at whom?", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()