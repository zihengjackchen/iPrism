from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt


p1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
p2 = Polygon([(3, 3), (3, 7), (7, 7), (7, 3)])
p3 = Polygon([(5, 3), (5, 7), (9, 7), (9, 3)])
p4 = Polygon([(15, 3), (15, 7), (19, 7), (19, 3)])


m = unary_union([p2, p3, p4])

print(m)

x2, y2 = m.exterior.xy
plt.plot(x2, y2)
# plt.show()
