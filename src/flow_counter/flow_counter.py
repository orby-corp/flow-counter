import cv2
from ultralytics import YOLO
from tqdm import tqdm

from flow_counter.utils import Point, intersect

class FlowCounter:
    def __init__(self, model_path: str = "yolo11n.pt"):
        """
        Initialize the flow counter with a given YOLO model.

        :param model_path: Path to the YOLO model file.
        """
        self.model = YOLO(model_path)

    def object_counts(self, input_path: str, output_path: str, line: tuple[Point, Point]) -> None:
        """
        Count objects crossing a line in a video.

        :param input_path: Path to the input video file.
        :param output_path: Path to the output video file (annotated).
        :param line: A tuple of two points defining the line ((x1, y1), (x2, y2))
        """
        cap = cv2.VideoCapture(input_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (frame_width, frame_height),
        )

        counter = 0
        counted_ids = set()
        
        with tqdm(total=total_frames, desc=f"Processing {input_path}") as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                results = self.model.track(frame, persist=True, verbose=False)

                # [x1, y1, x2, y2] format
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if results[0].boxes.id is None:
                    ids = [-1] * len(boxes)
                else:
                    ids = results[0].boxes.id.cpu().numpy()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    box_id = int(ids[i])
                    if intersect((x1, y1), (x2, y2), line[0], line[1]) and box_id != -1 and box_id not in counted_ids:
                        counter += 1
                        counted_ids.add(box_id)

                annotated_frame = results[0].plot()
                cv2.line(annotated_frame, line[0], line[1], (0, 255, 255), 3)
                cv2.putText(annotated_frame, str(counter), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)

                out.write(annotated_frame)
                pbar.update(1)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
