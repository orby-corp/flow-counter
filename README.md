# flow-counter
This is a package that counts the number of objects crossing a specified line in a video.

## Installation
```bash
git clone https://github.com/mhal-teddy/flow-counter.git
cd flow-counter
pip install -e .
```

## Python
```python
from flow_counter import FlowCounter

fc = FlowCounter("yolo11n.pt")
fc.object_counts("input.mp4", "output.mp4", ((100, 300), (800, 300)))
```

## License

This project is licensed under the terms of the GNU Affero General Public License v3.0 (AGPL-3.0).  
For details, see the [LICENSE](./LICENSE) file.
