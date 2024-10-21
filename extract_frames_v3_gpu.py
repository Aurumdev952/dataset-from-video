import concurrent.futures
import os

import cv2
import numpy as np
import openvino as ov
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ASSETS


# Custom OpenVINO model class for YOLOv8 inference on GPU
class RunIntelModel:
    def __init__(self) -> None:
        self.args = dict(
            model="yolo11n.pt",
            source=ASSETS,
            imgsz=(640, 640),
            save=False,
            verbose=False,
        )
        self.predictor = DetectionPredictor(overrides=self.args)
        self.predictor.setup_model("yolo11n.pt")
        self.predictor.model.pt = False
        self.det_model_path = (
            "yolo11n_openvino_model/yolo11n.xml"  # Path to OpenVINO IR model
        )
        self.openvino_model = self.setup_model()
        self.predictor.inference = self.infer

    def setup_model(self):
        core = ov.Core()
        det_ov_model = core.read_model(self.det_model_path)
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        det_ov_model.reshape({0: [1, 3, 640, 640]})
        det_compiled_model = core.compile_model(det_ov_model, "GPU", ov_config)
        return det_compiled_model

    def infer(self, *args):
        result = self.openvino_model(args)
        return torch.from_numpy(result[0])

    def predict(self, raw_img):
        res = self.predictor(raw_img)
        if res[0].boxes.shape[0] > 0:
            return res[0].boxes.conf[0], res[0].boxes.xywh
        else:
            return 0, [0, 0, 0, 0]

    def __call__(self, img):
        return self.predict(raw_img=img)


# Directory to save extracted frames
output_dir = "./extracted_frames/"
os.makedirs(output_dir, exist_ok=True)

batch_size = 8  # Number of frames processed at once
blur_threshold = 35  # Threshold for determining blurry images
near_vehicle_size_threshold = (
    0.15  # Minimum percentage of the frame covered by the vehicle
)

model = RunIntelModel()  # Initialize OpenVINO-integrated YOLO model


def is_clear_frame(frame):
    """Check if the frame is not blurry using the Laplacian variance method."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > blur_threshold


def is_vehicle_near(box, frame):
    """Check if the vehicle is near by calculating the size of the bounding box relative to the frame."""
    x_min, y_min, x_max, y_max = box
    vehicle_area = (x_max - x_min) * (y_max - y_min)
    frame_area = frame.shape[0] * frame.shape[1]
    return (vehicle_area / frame_area) > near_vehicle_size_threshold


def process_video_chunk(video_path, start_frame, end_frame, chunk_id):
    """Process a video chunk and extract frames containing clear and near vehicles."""
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame
    total_frames_in_chunk = end_frame - start_frame

    with tqdm(
        total=total_frames_in_chunk,
        desc=f"Processing Chunk {chunk_id + 1}",
        unit="frame",
    ) as pbar:
        frame_batch = []
        frame_indices = []

        while video.isOpened() and frame_count < end_frame:
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1
            frame_batch.append(frame)
            frame_indices.append(frame_count)

            if len(frame_batch) == batch_size or frame_count == end_frame:
                process_batch(frame_batch, frame_indices, chunk_id)
                frame_batch.clear()
                frame_indices.clear()

            pbar.update(len(frame_batch))

    video.release()
    print(f"Chunk {chunk_id + 1} processed frames from {start_frame} to {end_frame}.")


def process_batch(frames, frame_indices, chunk_id):
    """Run OpenVINO-integrated YOLO on a batch of frames and save frames containing clear and near vehicles."""
    results = [model(frame) for frame in frames]  # Inference on each frame

    for i, result in enumerate(results):
        frame = frames[i]
        frame_count = frame_indices[i]

        # If the result is a detection with confidence
        if result[0] > 0:
            box = result[1]  # Bounding box coordinates (tensor)

            # Convert the tensor to a numpy array and extract individual values
            box = box.cpu().numpy().flatten()  # Flatten to ensure it's a 1D array
            x_min, y_min, w, h = map(
                int, box[:4]
            )  # Extract first 4 values (x, y, width, height)
            x_max, y_max = x_min + w, y_min + h

            if is_clear_frame(frame) and is_vehicle_near(
                (x_min, y_min, x_max, y_max), frame
            ):
                frame_name = f"chunk_{chunk_id}_frame_{frame_count:04d}.jpg"
                output_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(output_path, frame)


def split_video_and_process(video_path, num_chunks=5):
    """Split the video into chunks and process them in parallel."""
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    frames_per_chunk = total_frames // num_chunks
    frame_ranges = [
        (i * frames_per_chunk, (i + 1) * frames_per_chunk) for i in range(num_chunks)
    ]
    frame_ranges[-1] = (
        frame_ranges[-1][0],
        total_frames,
    )  # Adjust last chunk to the end

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for chunk_id, (start_frame, end_frame) in enumerate(frame_ranges):
            futures.append(
                executor.submit(
                    process_video_chunk, video_path, start_frame, end_frame, chunk_id
                )
            )

        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    video_path = "input-video.mp4"
    split_video_and_process(video_path, num_chunks=10)
