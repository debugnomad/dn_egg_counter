import cv2
import numpy as np
from ultralytics import YOLO
#====================================+#
## Predict
model = YOLO('./runs/detect/train/weights/best.pt')  # load a custom model

# Define path to video file
source = './videos/egg_test_2.mp4'
output_dir = './videos/output_2.mp4'

boundary_line = 250 + 250
boundary_line2 = 300 + 250
color_in = (0, 255, 0)  # Green color
color_out =(0,0,255)
color_done = (255,0,0)
thickness = 2
id_set = set()

cap = cv2.VideoCapture(source)
video_writer = cv2.VideoWriter(output_dir, cv2.VideoWriter_fourcc(*"mp4v"),
                               cap.get(cv2.CAP_PROP_FPS), (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                           round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
# video_writer = cv2.VideoWriter(output_dir, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (378,600))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # frame = frame[250:,100:,:]
        annotated_frame=frame.copy()

        # Run YOLOv8 inference on the frame
        # results = model(frame,conf=0.5)
        results = model.track(frame, persist=True,conf=0.5)
        # Draw boundary line
        cv2.line(annotated_frame, (0, boundary_line), (annotated_frame.shape[1], boundary_line), (0, 255, 0), 2)
        cv2.line(annotated_frame, (0, boundary_line2), (annotated_frame.shape[1], boundary_line2), (0, 255, 0), 2)
        bboxs,ids = results[0].boxes.data,results[0].boxes.id
        if ids is None:
            continue
        for bbox,id in zip(bboxs,ids):
            x1, y1, x2, y2= bbox[0],bbox[1],bbox[2],bbox[3]
            # x1, y1, x2, y2 ,_,_= bbox
            if y1 < boundary_line :
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color_out, thickness)
                cv2.putText(annotated_frame, f'Count: {len(id_set)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            elif y1 > boundary_line and y1 < boundary_line2:
                if not id_set.__contains__(id.item()):
                    id_set.add(id.item())
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color_in, thickness)
                # cv2.putText(annotated_frame, f'{id.item()}', (int(x2) - 20, int(y2) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color_in, 2)
                cv2.putText(annotated_frame, f'Count: {len(id_set)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            elif y1 > boundary_line2:
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color_done, thickness)
                cv2.putText(annotated_frame, f'Count: {len(id_set)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        video_writer.write(annotated_frame.astype(np.uint8))
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

