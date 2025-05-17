import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from flow_counter.union_find import DictUnionFind
from flow_counter.utils import Point, VEHICLE_NAMES, intersect, compute_iou

class FlowCounter:
    def __init__(self, model_path: str = "yolo11n.pt"):
        """
        Initialize the flow counter with a given YOLO model.

        :param model_path: Path to the YOLO model file.
        """
        self.model = YOLO(model_path)
        self.uf = DictUnionFind()
        self._reset()

    def _reset(self):
        # Set of already-counted object IDs.
        self.counted_ids: set[int] = set()

        # Dictionary to store class-wise counts.
        self.cls_counts: dict[str, int] = {}

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
        xyxys: np.ndarray,
        ids: list[int],
        classes: np.ndarray,
        line: tuple[Point, Point],
    ) -> int:
        """
        Count how many objects crossed a specific line.

        :param xyxys: Array of bounding boxes [[x1, y1, x2, y2], ...]
        :param ids: List of object IDs correspoinding to the boxes.
        :param classes: Class IDs corresponding to the boxes.
        :param line: Line represented by two points (start, end).
        :return Number of new objects crossing the line.
        """
        count = 0
        for xyxy, box_id, cls_id in zip(xyxys, ids, classes):
            x1, y1, x2, y2 = map(int, xyxy)
            root_id = self.uf.find(box_id)
            class_name = self.model.names[cls_id]

            if box_id == -1 or root_id in self.counted_ids:
                continue

            if class_name not in VEHICLE_NAMES:
                continue

            if intersect((x1, y1), (x2, y2), line[0], line[1]):
                count += 1
                self.counted_ids.add(root_id)
                self.cls_counts[class_name] = self.cls_counts.get(class_name, 0) + 1
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
        cv2.putText(frame, str(counter), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
        return frame

    def object_counts(self, input_path: str, output_path: str, line: tuple[Point, Point]) -> None:
        """
        Count objects crossing a line in a video.

        :param input_path: Path to the input video file.
        :param output_path: Path to the output video file (annotated).
        :param line: A tuple of two points defining the line ((x1, y1), (x2, y2))
        """
        self._reset()
        cap, total_frames, frame_size = self._open_video(input_path)

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            frame_size,
        )

        counter = 0
        
        with tqdm(total=total_frames, desc=f"Processing {input_path}") as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                results = self.model.track(frame, persist=True, verbose=False)
                boxes = results[0].boxes

                # [x1, y1, x2, y2] format
                xyxys = boxes.xyxy.cpu().numpy()
                if boxes.id is not None:
                    ids = np.round(boxes.id.cpu().numpy()).astype(int).tolist()
                else:
                    ids = [-1] * len(boxes)
                classes = boxes.cls.cpu().numpy()

                # Updated Non-Maximum Suppression
                for i in range(len(xyxys)):
                    for j in range(i + 1, len(xyxys)):
                        iou = compute_iou(xyxys[i], xyxys[j])
                        if iou >= 0.5:
                            self.uf.unite(ids[i], ids[j])

                counter += self._count_crossing_objects(xyxys, ids, classes, line)

                annotated_frame = results[0].plot()
                annotated_frame = self._annotate_frame(annotated_frame, line, counter)

                out.write(annotated_frame)
                pbar.update(1)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
