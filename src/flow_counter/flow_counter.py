import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from flow_counter.utils import Point, intersect

class FlowCounter:
    def __init__(self, model_path: str = "yolo11n.pt"):
        """
        Initialize the flow counter with a given YOLO model.

        :param model_path: Path to the YOLO model file.
        """
        self.model = YOLO(model_path)

    def _open_video(self, input_path: str) -> tuple[cv2.VideoCapture, int, tuple[int, int]]:
        """
        Open a video file and retrieve basic metadata.

        :param input_path: Path to the input video file.
        :return: Tuple of (VideoCapture object, total frame count, frame size (width, height))
        """
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap, total_frames, (frame_width, frame_height)
    
    def _count_crossing_objects(
        self,
        boxes: np.ndarray,
        ids: np.ndarray,
        line: tuple[Point, Point],
        counted_ids: set[int],
    ) -> int:
        """
        Count how many objects crossed a specific line.

        :param boxes: Array of bounding boxes [[x1, y1, x2, y2], ...]
        :param ids: Array of object IDs correspoinding to the boxes.
        :param line: Line represented by two points (start, end).
        :param counted_ids: Set of already-counted object IDs.
        :return Number of new objects crossing the line.
        """
        count = 0
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            box_id = int(ids[i])

            if intersect((x1, y1), (x2, y2), line[0], line[1]) and box_id != -1 and box_id not in counted_ids:
                count += 1
                counted_ids.add(box_id)
        return count
    
    def _annotate_frame(self, frame: np.ndarray, line: tuple[Point, Point], counter: int) -> np.ndarray:
        """
        Draw the counting line and current count on the frame.

        :param frame: The current frame to annotate.
        :param line: The counting line.
        :param counter: The number of objects counted so far.
        :return: Annotated frame.
        """
        cv2.line(frame, line[0], line[1], (0, 255, 255), 3)
        cv2.putText(frame, str(counter), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
        return frame

    def object_counts(self, input_path: str, output_path: str, line: tuple[Point, Point]) -> None:
        """
        Count objects crossing a line in a video.

        :param input_path: Path to the input video file.
        :param output_path: Path to the output video file (annotated).
        :param line: A tuple of two points defining the line ((x1, y1), (x2, y2))
        """
        cap, total_frames, frame_size = self._open_video(input_path)

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            frame_size,
        )

        counter = 0
        counted_ids: set[int] = set()
        
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

                counter += self._count_crossing_objects(boxes, ids, line, counted_ids)

                annotated_frame = results[0].plot()
                annotated_frame = self._annotate_frame(annotated_frame, line, counter)

                out.write(annotated_frame)
                pbar.update(1)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
