# flow-counter
This is a package that counts the number of objects crossing a specified line in a video.

## Installation
```bash
git clone https://github.com/orby-corp/flow-counter.git
cd flow-counter
pip install -e .
# pip install -e .[dev]  # For development
```

## Python
```python
from flow_counter.flow_counter import FlowCounter

# Initialize FlowCounter with YOLO model
fc = FlowCounter("yolo11n.pt")

# Define two counting lines for each area name
# Each entry in the line_map contains a pair of lines ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4))
line_map = {
    "road": (((100, 300), (800, 300)), ((100, 400), (800, 400)))
}

# Perform counting
fc.object_counts("input.mp4", "output.mp4", line_map)

# Display results
# Example output: {"person": {"road": 2}, "car": {"road": 3}, "motorcycle": {}, "bus": {}, "truck": {}}
print(fc.cls_counts)
```

## License

This project is licensed under the terms of the GNU Affero General Public License v3.0 (AGPL-3.0).  
For details, see the [LICENSE](./LICENSE) file.
