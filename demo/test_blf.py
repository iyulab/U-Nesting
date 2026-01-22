#!/usr/bin/env python3
"""Test BLF placement bounds."""
import json
import urllib.request
import math

# Demo data
data = {
    "input": {
        "name": "demo",
        "items": [
            {"id": 0, "demand": 2, "allowed_orientations": [0,90,180,270],
             "shape": {"type": "simple_polygon", "data": [[0,0],[180,0],[195,15],[200,50],[200,150],[195,185],[180,200],[20,200],[5,185],[0,150],[0,50],[5,15],[0,0]]}},
            {"id": 1, "demand": 4, "allowed_orientations": [0,45,90,135],
             "shape": {"type": "simple_polygon", "data": [[60,0],[85,7],[104,25],[118,50],[120,60],[118,70],[104,95],[85,113],[60,120],[35,113],[16,95],[2,70],[0,60],[2,50],[16,25],[35,7],[60,0]]}},
            {"id": 2, "demand": 6, "allowed_orientations": [0,90,180,270],
             "shape": {"type": "simple_polygon", "data": [[0,0],[80,0],[80,20],[20,20],[20,80],[0,80],[0,0]]}},
            {"id": 3, "demand": 6, "allowed_orientations": [0,90,180,270],
             "shape": {"type": "simple_polygon", "data": [[0,0],[70,0],[0,70],[0,0]]}},
            {"id": 4, "demand": 4, "allowed_orientations": [0,90],
             "shape": {"type": "simple_polygon", "data": [[0,0],[120,0],[120,60],[0,60],[0,0]]}},
            {"id": 5, "demand": 8, "allowed_orientations": [0,60,120],
             "shape": {"type": "simple_polygon", "data": [[15,0],[45,0],[60,26],[45,52],[15,52],[0,26],[15,0]]}},
            {"id": 6, "demand": 4, "allowed_orientations": [0,90,180,270],
             "shape": {"type": "simple_polygon", "data": [[0,0],[90,0],[90,12],[55,12],[55,60],[35,60],[35,12],[0,12],[0,0]]}},
            {"id": 7, "demand": 3, "allowed_orientations": [0,90],
             "shape": {"type": "simple_polygon", "data": [[0,10],[10,0],[70,0],[80,10],[80,70],[70,80],[10,80],[0,70],[0,10]]}},
            {"id": 8, "demand": 13, "allowed_orientations": [0,45,90,135,180,225,270,315],
             "shape": {"type": "simple_polygon", "data": [[50,5],[65,15],[77,18],[80,32],[95,50],[80,68],[77,82],[65,85],[50,95],[35,85],[23,82],[20,68],[5,50],[20,32],[23,18],[35,15],[50,5]]}}
        ],
        "strip_width": 500,
        "strip_height": 500
    },
    "strategy": "blf"
}

# Geometry shapes for bounds checking
shapes = {
    0: [[0,0],[180,0],[195,15],[200,50],[200,150],[195,185],[180,200],[20,200],[5,185],[0,150],[0,50],[5,15],[0,0]],
    1: [[60,0],[85,7],[104,25],[118,50],[120,60],[118,70],[104,95],[85,113],[60,120],[35,113],[16,95],[2,70],[0,60],[2,50],[16,25],[35,7],[60,0]],
    2: [[0,0],[80,0],[80,20],[20,20],[20,80],[0,80],[0,0]],
    3: [[0,0],[70,0],[0,70],[0,0]],
    4: [[0,0],[120,0],[120,60],[0,60],[0,0]],
    5: [[15,0],[45,0],[60,26],[45,52],[15,52],[0,26],[15,0]],
    6: [[0,0],[90,0],[90,12],[55,12],[55,60],[35,60],[35,12],[0,12],[0,0]],
    7: [[0,10],[10,0],[70,0],[80,10],[80,70],[70,80],[10,80],[0,70],[0,10]],
    8: [[50,5],[65,15],[77,18],[80,32],[95,50],[80,68],[77,82],[65,85],[50,95],[35,85],[23,82],[20,68],[5,50],[20,32],[23,18],[35,15],[50,5]]
}

# Piece ID to shape ID mapping (based on demand expansion)
# piece 0-1: shape 0 (demand 2)
# piece 2-5: shape 1 (demand 4)
# piece 6-11: shape 2 (demand 6)
# piece 12-17: shape 3 (demand 6)
# piece 18-21: shape 4 (demand 4)
# piece 22-29: shape 5 (demand 8)
# piece 30-33: shape 6 (demand 4)
# piece 34-36: shape 7 (demand 3)
# piece 37-49: shape 8 (demand 13)
def get_shape_id(piece_id):
    if piece_id < 2: return 0
    if piece_id < 6: return 1
    if piece_id < 12: return 2
    if piece_id < 18: return 3
    if piece_id < 22: return 4
    if piece_id < 30: return 5
    if piece_id < 34: return 6
    if piece_id < 37: return 7
    return 8

def rotate_point(x, y, angle):
    """Rotate point by angle (radians)."""
    c, s = math.cos(angle), math.sin(angle)
    return x*c - y*s, x*s + y*c

def get_bounds_at_rotation(shape, rotation):
    """Get AABB bounds of shape at given rotation."""
    rotated = [rotate_point(x, y, rotation) for x, y in shape]
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    return min(xs), min(ys), max(xs), max(ys)

# Send request
req = urllib.request.Request(
    'http://localhost:8888/api/optimize',
    data=json.dumps(data).encode(),
    headers={'Content-Type': 'application/json'}
)

with urllib.request.urlopen(req, timeout=120) as resp:
    result = json.loads(resp.read())

placements = result.get('placements', [])
strip_width = 500
strip_height = 500

print(f"Total placements: {len(placements)}")
print(f"Strip size: {strip_width}x{strip_height}")
print()

# Check each placement
violations = []
for p in placements:
    x, y = p['position']
    rotation = p['rotation']
    piece_id = int(p['geometry_id'].replace('piece_', ''))
    shape_id = get_shape_id(piece_id)
    shape = shapes[shape_id]

    # Get bounds at this rotation
    min_x, min_y, max_x, max_y = get_bounds_at_rotation(shape, rotation)

    # Calculate actual bounds when placed
    actual_min_x = x + min_x
    actual_min_y = y + min_y
    actual_max_x = x + max_x
    actual_max_y = y + max_y

    # Determine strip
    strip_idx = int(x // strip_width)
    local_x = x - strip_idx * strip_width
    local_actual_min_x = local_x + min_x
    local_actual_max_x = local_x + max_x

    # Check bounds within strip
    out_of_bounds = False
    reasons = []
    if local_actual_min_x < 0:
        out_of_bounds = True
        reasons.append(f"left={local_actual_min_x:.1f}<0")
    if local_actual_max_x > strip_width:
        out_of_bounds = True
        reasons.append(f"right={local_actual_max_x:.1f}>{strip_width}")
    if actual_min_y < 0:
        out_of_bounds = True
        reasons.append(f"bottom={actual_min_y:.1f}<0")
    if actual_max_y > strip_height:
        out_of_bounds = True
        reasons.append(f"top={actual_max_y:.1f}>{strip_height}")

    if out_of_bounds:
        violations.append({
            'piece': p['geometry_id'],
            'strip': strip_idx + 1,
            'local_x': local_x,
            'y': y,
            'rotation_deg': rotation * 180 / math.pi,
            'shape_id': shape_id,
            'reasons': reasons
        })

if violations:
    print(f"VIOLATIONS FOUND: {len(violations)} items outside bounds")
    print("-" * 60)
    for v in violations:
        print(f"{v['piece']}: strip={v['strip']}, pos=({v['local_x']:.1f}, {v['y']:.1f}), rot={v['rotation_deg']:.0f}°, shape={v['shape_id']}")
        print(f"  Reasons: {', '.join(v['reasons'])}")
else:
    print("All placements within bounds!")
