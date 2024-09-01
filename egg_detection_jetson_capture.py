# egg_detection_jetson_capture.py, this code uses jetson_utils library to capture video and a custom yolo model to detect eggs from a camera source.
# Copyright (C) 2024 DEBUG NOMAD SLU, www.debugnomad.com

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ultralytics import YOLO
import torch
import jetson_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}") # cuda
model = YOLO('./yolo_egg.pt').to(device) # load a custom model

boundary_line = 250 + 250
boundary_line2 = 300 + 250
id_set = set()

counter = None

# Initialize video capture using jetson.utils
source = 'v4l2:///dev/video0'  # Adjust according to your USB camera device
cap = jetson_utils.videoSource(source)
output = jetson_utils.videoOutput()
img = cap.Capture()

# Loop through the video frames
while cap.IsStreaming():
    # Read a frame from the video
    if img:
        # Convert the image from jetson.utils to a numpy array
        np_img = jetson_utils.cudaToNumpy(img)

        # Convert RGBA to RGB
        np_img_rgb = np_img[:, :, :3]

        # Run YOLOv8 inference on the frame
        results = model.track(np_img_rgb, persist=True, conf=0.5, device=device)
        
        bboxs, ids = results[0].boxes.data, results[0].boxes.id
        if ids is not None:
            for bbox, id in zip(bboxs, ids):
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                if y1 > boundary_line and y1 < boundary_line2:
                    id_set.add(id.item())
                    counter = len(id_set)
                    print(counter)
        frame_overlay = results[0].plot()
        # cv2.imshow("YOLOv8 Inference", frame_overlay)
        output.Render(jetson_utils.cudaFromNumpy(frame_overlay))

    img = cap.Capture()

# Release the video capture object and close the display window
cap.Close()