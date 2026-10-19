# RD-DETR documents can be found here: https://rfdetr.roboflow.com/latest/

import threading
import time
import os
import supervision as sv
import cv2
from rfdetr import RFDETRSmall

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"
model = RFDETRSmall(pretrain_weights="runs/drone_v1/checkpoint_best_total.pth")

box_annotator = sv.BoxAnnotator(
    thickness=3
)
label_annotator = sv.LabelAnnotator(
    text_scale=1.5,
    text_thickness=3
)


class FrameReader:
    def __init__(self, source):
        self.source = source
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
 
    def _read_loop(self):
        while self.running:
            video_capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            while self.running and video_capture.isOpened():
                success, frame_bgr = video_capture.read()
                if not success:
                    break
                self.frame = frame_bgr
                self.frame_id += 1
            video_capture.release()
            time.sleep(1)  # stream not started yet, or it dropped; try again
 
    def stop(self):
        self.running = False


def capture_video():
    video_source = "test_videos/turtle_burrow.mp4"
    video_capture = cv2.VideoCapture(video_source)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_source}")

    cv2.namedWindow("RF-DETR Video", cv2.WINDOW_NORMAL) # initialize window
    cv2.resizeWindow("RF-DETR Video", 512, 512)         # resize window
    print("Window opening, press 'q' to exit window")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break


        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # convert to RGB format
        detections = model.predict(frame_rgb, threshold=0.3)

        labels = [
            f"{model.class_names[class_id]} {confidence:.2%}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        annotated_frame = box_annotator.annotate(frame_bgr, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR Video", annotated_frame)        # display window

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def capture_live():
    reader = FrameReader("udp://127.0.0.1:5000?fifo_size=5000000&overrun_nonfatal=1")
    last_frame_id = 0

    cv2.namedWindow("RF-DETR Video", cv2.WINDOW_NORMAL) # initialize window
    cv2.resizeWindow("RF-DETR Video", 512, 512)         # resize window
    print("Window opening, press 'q' to exit window")

    while True:
            # nothing new to process yet: keep the window responsive and wait
            if reader.frame is None or reader.frame_id == last_frame_id:
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue

            last_frame_id = reader.frame_id
            frame_bgr = reader.frame.copy()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # convert to RGB format
            detections = model.predict(frame_rgb, threshold=0.3)

            labels = [
                        f"{model.class_names[class_id]} {confidence:.2%}"
                        for class_id, confidence in zip(detections.class_id, detections.confidence)
                    ]
            
            annotated_frame = box_annotator.annotate(frame_bgr, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

            cv2.imshow("RF-DETR Video", annotated_frame)        # display window

    reader.stop()
    cv2.destroyAllWindows()

def main():
    capture_live()
    # will add another function later, likely

if __name__ == "__main__":
    main()
