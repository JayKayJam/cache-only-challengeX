# RD-DETR documents can be found here: https://rfdetr.roboflow.com/latest/

import supervision as sv
import cv2
from rfdetr import RFDETRSmall

model = RFDETRSmall(pretrain_weights="runs/drone_v1/checkpoint_best_total.pth")

box_annotator = sv.BoxAnnotator(
    thickness=3
)
label_annotator = sv.LabelAnnotator(
    text_scale=1.5,
    text_thickness=3
)

def capture_video():
    video_source = "test_videos/turtle_burrow.mp4"
    video_capture = cv2.VideoCapture(video_source)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_source}")

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

        cv2.namedWindow("RF-DETR Video", cv2.WINDOW_NORMAL) # initialize window
        cv2.imshow("RF-DETR Video", annotated_frame)        # display window
        cv2.resizeWindow("RF-DETR Video", 512, 512)         # resize window

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def main():
    capture_video()
    # will add another function later, likely

if __name__ == "__main__":
    main()
