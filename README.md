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

# Number of counted objects that crossed line2 before line1 (reverse direction), per line name.
# Included in the counts above; use this to spot-check how much reverse-direction traffic
# (e.g. oncoming lane) is being counted. Example output: {"road": 1}
print(fc.reverse_crossings)

# Human-readable summary of everything object_counts() tracked internally,
# including which frame each ID merge / line crossing happened on.
stats = fc.get_statistics()
print(stats["cls_counts"])          # same content as fc.cls_counts
print(stats["reverse_crossings"])   # same content as fc.reverse_crossings
print(stats["first_crossed_side"])  # {"road": {"5": "line1"}}
print(stats["merged_id_groups"])    # tracker IDs merged into one object, e.g. {"5": ["5", "7"]}
print(stats["merge_timeline"])      # ["frame 8: object 7 と object 5 を統合（統合後のroot: 5）"]
print(stats["crossing_timeline"])   # ["frame 12: car (id=5) が road を通過（順方向）"]
```

## License

This project is licensed under the terms of the GNU Affero General Public License v3.0 (AGPL-3.0).  
For details, see the [LICENSE](./LICENSE) file.
