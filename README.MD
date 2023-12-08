# RGBA color search with Cython

## pip install cythonrgbasearch

### Tested against Windows / Python 3.11 / Anaconda


## Cython (and a C/C++ compiler) must be installed



```python
from cythonrgbasearch import find_rgba_colors, find_rgb_colors
import cv2
import numpy as np

data = cv2.imread(r"C:\Users\hansc\Downloads\pexels-alex-andrews-2295744.jpg")
# 4525 x 6623 x 3 picture https://www.pexels.com/pt-br/foto/foto-da-raposa-sentada-no-chao-2295744/
data = np.ascontiguousarray(np.dstack([data, np.full(data.shape[:2], 0, dtype=np.uint8)]))
colors1 = np.array(
    [(66, 71, 69), (62, 67, 65), (144, 155, 153), (52, 57, 55), (127, 138, 136), (53, 58, 56), (51, 56, 54),
     (32, 27, 18), (24, 17, 8), ], dtype=np.uint8)
colors1 = np.array([list((x.tolist())) + [0] for x in colors1], dtype=np.uint8)
ocol = np.array([[255, 255, 255, 0]], dtype=np.uint8)
r1 = find_rgba_colors(pic=data, colors=colors1, dummy_alpha=0)
r2 = find_rgba_colors(pic=data, colors=ocol, dummy_alpha=0)
r3 = find_rgb_colors(pic=data, colors=colors1)
r4 = find_rgb_colors(pic=data, colors=ocol)

# r1
# Out[4]:
# rec.array([(136, 138, 127, 0,    0,   38), ( 69,  71,  66, 0,    0, 4522),
#            ( 65,  67,  62, 0,    0, 4523), ...,
#            (  8,  17,  24, 0, 4524, 6620), (  8,  17,  24, 0, 4524, 6621),
#            (  8,  17,  24, 0, 4524, 6622)],
#           dtype=[('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'), ('x', '<i8'), ('y', '<i8')])
# r2
# Out[5]:
# rec.array([(255, 255, 255, 0,  568, 5021), (255, 255, 255, 0,  586, 3064),
#            (255, 255, 255, 0,  612, 2812), ...,
#            (255, 255, 255, 0, 3752,  805), (255, 255, 255, 0, 3752,  806),
#            (255, 255, 255, 0, 3775, 1291)],
#           dtype=[('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'), ('x', '<i8'), ('y', '<i8')])
# r3
# Out[6]:
# rec.array([(136, 138, 127,    0,   38), ( 69,  71,  66,    0, 4522),
#            ( 65,  67,  62,    0, 4523), ..., (  8,  17,  24, 4524, 6620),
#            (  8,  17,  24, 4524, 6621), (  8,  17,  24, 4524, 6622)],
#           dtype=[('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('x', '<i8'), ('y', '<i8')])
# r4
# Out[7]:
# rec.array([(255, 255, 255,  568, 5021), (255, 255, 255,  586, 3064),
#            (255, 255, 255,  612, 2812), ..., (255, 255, 255, 3752,  805),
#            (255, 255, 255, 3752,  806), (255, 255, 255, 3775, 1291)],
#           dtype=[('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('x', '<i8'), ('y', '<i8')])


```