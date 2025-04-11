from math import radians
import shapely.geometry as sgeo

from kinematic_model import KinematicBicycleModel
from libs import CarDescription

import numpy as np
import math

import matplotlib.pyplot as plt

import os
import concurrent.futures

import time

import numpy as np
from scipy.spatial import Delaunay
from shapely.ops import cascaded_union, polygonize


def alpha_shape(points, alpha):
    """
    Compute the alpha shape of a set of points.
    
    Parameters:
    - points: list of (x,y) pairs representing the points.
    - alpha: alpha value.
    
    Returns:
    - Geometries representing the alpha shape.
    """
    
    # Delaunay triangulation
    tri = Delaunay(points)
    
    # Triangle vertices
    triangles = points[tri.vertices]
    
    # Alpha criterion: Circumradius must be smaller than alpha
    a = triangles[:,0,:]
    b = triangles[:,1,:]
    c = triangles[:,2,:]
    
    # Lengths of triangle sides
    a_b = np.linalg.norm(a - b, axis=1)
    b_c = np.linalg.norm(b - c, axis=1)
    c_a = np.linalg.norm(c - a, axis=1)
    
    # Semiperimeter
    s = (a_b + b_c + c_a) / 2.0
    
    # Area
    area = np.sqrt(s*(s-a_b)*(s-b_c)*(s-c_a))
    
    # Circumradius
    circum_r = a_b * b_c * c_a / (4.0 * area)
    
    # Filter triangles
    valid = circum_r < alpha
    
    # Create the edges
    edges = []
    for i in range(triangles.shape[0]):
        if valid[i]:
            edge1 = [triangles[i, 0], triangles[i, 1]]
            edge2 = [triangles[i, 1], triangles[i, 2]]
            edge3 = [triangles[i, 2], triangles[i, 0]]
            edges.append(edge1)
            edges.append(edge2)
            edges.append(edge3)

    # Create a MultiLineString from the edges and extract polygons
    m = sgeo.MultiLineString(edges)
    triangles = list(polygonize(m))

    return cascaded_union(triangles)
    


class Car:
    def __init__(self, init_x, init_y, init_v, init_yaw, delta_time):

        # Model parameters
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.velocity = init_v
        self.delta_time = delta_time
        self.time = 0.0
        self.wheel_angle = 0.0
        self.angular_velocity = 0.0
        self.acceleration = 0.0
        max_steer = radians(50)     # 33 degrees => radians
        wheelbase = 2.96

        # Car parameters
        overall_length = 4.97
        overall_width = 1.964
        tyre_diameter = 0.4826
        tyre_width = 0.265
        axle_track = 1.7
        rear_overhang = 0.5 * (overall_length - wheelbase)

        self.kinematic_bicycle_model = KinematicBicycleModel(wheelbase, max_steer, self.delta_time)
        self.description = CarDescription(overall_length, overall_width, rear_overhang, tyre_diameter, tyre_width, axle_track, wheelbase)

    def plot_car(self):
        return self.description.plot_car(self.x, self.y, self.yaw, self.wheel_angle)

    def drive(self, acceleration, steerAngle):
        self.acceleration = acceleration
        self.wheel_angle = steerAngle
        self.x, self.y, self.yaw, self.velocity, _, _ = self.kinematic_bicycle_model.update(self.x, self.y, self.yaw, self.velocity, self.acceleration, self.wheel_angle)

    def setStates(self, init_x, init_y, init_v, init_yaw):
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.velocity = init_v



# https://www.geeksforgeeks.org/check-if-any-point-overlaps-the-given-circle-and-rectangle/
def checkOverlap(R, Xc, Yc, X1, Y1, X2, Y2):
    # Find the nearest point on the
    # rectangle to the center of
    # the circle
    Xn = max(X1, min(Xc, X2))
    Yn = max(Y1, min(Yc, Y2))
     
    # Find the distance between the
    # nearest point and the center
    # of the circle
    Dx = Xn - Xc
    Dy = Yn - Yc
    return (Dx**2 + Dy**2) < R**2

def collided(egoCar, obstacles, xStep, yStep, radius):
    return any([collided_single(egoCar, obstacle, xStep, yStep, radius) for obstacle in obstacles])


def collided_single(egoCar, obstacle, xStep, yStep, radius):
    obstacle = np.array(obstacle)
    xmin = np.min(obstacle[:, 0])
    xmax = np.max(obstacle[:, 0])
    ymin = np.min(obstacle[:, 1])
    ymax = np.max(obstacle[:, 1])
    xminGrid = int(xmin / xStep)
    yminGrid = int(ymin / yStep)
    xmaxGrid = int(math.ceil(xmax / xStep))
    ymaxGrid = int(math.ceil(ymax / yStep))
    
    egoCarXGrid = int(round(egoCar.x / xStep))
    egoCarYGrid = int(round(egoCar.y / yStep))
    
    # if xminGrid <= egoCarXGrid <= xmaxGrid and yminGrid <= egoCarYGrid <= ymaxGrid:
    if checkOverlap(radius/xStep, egoCarXGrid, egoCarYGrid, xminGrid, yminGrid, xmaxGrid, ymaxGrid):
        return True
        
    return False

# obstacles: list of all vehicles in the form of polygons that are blocking the AV
# name: name of current plot
# missing: missing vehicle, used when calculating risk of a specific vehicle
# full_area: reachable area when all vehicles are present
# empty_area: reachable area when no vehicles are present
# areas: the reachable area of each vecicle in order
# folder: name of folder to save the plot in
# show: flag to show the plot when running
# save_file: flag to save all risk to a file, used when generating last step
def STI_demo(obstacles, name, missing = None, empty_area = 0, areas = [], folder = "", verbose = False):
    if verbose:
        start = time.time()
    
    if missing != None:
        missing_obstacle = obstacles[missing]
        obstacles = obstacles[:missing] + obstacles[missing+1:]
    
    timeDelta = 0.1
    totalTime = 1.6
    maxAcc = +8

    minAcc = -7
    maxSteerRight = 33
    initX, initY, initV, initYaw = 0, 0, 5, 0
    egoCar = Car(initX, initY, initV, initYaw, timeDelta)
    xStep = yStep = vStep = 0.15
    yawStep = radians(5)
    street_radius = 3.0

    # Setting radius of car
    radius = 0.9
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    a = radius * np.cos( theta )
    b = radius * np.sin( theta )
 

          

    if verbose:
        print(f"[{folder} - {name}] Init: {(time.time()-start)}s")
        start = time.time()


    steppingSet = {}
    steppingSet[-1] = {(int(initX / xStep), int(initY / yStep), int(initV / vStep), int(initYaw / yawStep)): 1}
    
    for t in range (int(np.ceil(totalTime / timeDelta))):
        steppingHisto = {}
        if t-1 not in steppingSet or steppingSet[t-1]=={}:
            print(f"[{folder} - {name}] Last time step did not produce any possible points!")
            break
        for initCondSet in steppingSet[t-1]:
            currInitXIdx, currInitYIdx, currInitVIdx, currInitYawIdx = initCondSet
            currInitX, currInitY, currInitV, currInitYaw = currInitXIdx * xStep, currInitYIdx * yStep, currInitVIdx * vStep, currInitYawIdx * yawStep
            for steer in [-maxSteerRight, 0, maxSteerRight]:
                for acc in [maxAcc]:
                    egoCar.setStates(currInitX, currInitY, currInitV, currInitYaw)
                    egoCar.drive(acc, radians(steer))
                    if egoCar.velocity >= 0 and not collided(egoCar, obstacles, xStep, yStep, radius) and -street_radius <= egoCar.y <= street_radius:
                        bin = (int(round(egoCar.x / xStep)), int(round(egoCar.y / yStep)), int(round(egoCar.velocity / vStep)), int(round(egoCar.yaw / yawStep)))
                        steppingHisto[bin] = 1
            
            steppingSet[t] = steppingHisto
        
   
    if verbose:
        print(f"[{folder} - {name}] Sampling points: {(time.time()-start)}s")
        start = time.time()

    all_points = set()

    for timeStampSet in steppingSet.values():
        for initCondSet in timeStampSet:
            all_points.add((initCondSet[0] * xStep, initCondSet[1] * yStep))

    if verbose:
        print("Total number of points:", len(all_points))

    res = 0.0
    hull_pts = [0, 0]
    
    alpha = 1.4

    all_points_np = np.array(list(all_points))
    hull  = alpha_shape(all_points_np, alpha)


    adjusted = False
    
    
    # Polygon
    if isinstance(hull, sgeo.Polygon):
        hull_pts = hull.exterior.coords.xy
        res = hull.area
        for obstacle in obstacles:
            if hull.contains(sgeo.Point(obstacle[0])) and hull.contains(sgeo.Point(obstacle[1])) \
            and hull.contains(sgeo.Point(obstacle[2])) and hull.contains(sgeo.Point(obstacle[3])):
               rect = sgeo.Polygon()
               res -= (rect.area + rect.length * radius + 4 * radius * radius)
               adjusted = True
    
    # Line
    elif isinstance(hull, sgeo.LineString):
        hull_pts = hull.convex_hull.xy

    # Multipolygon
    elif isinstance(hull, sgeo.multipolygon.MultiPolygon):
        hull_pts = []
        res = hull.area
        for geom in hull.geoms:
            hull_pts.append((geom.exterior.xy[0], geom.exterior.xy[1]))
            for obstacle in obstacles:
                if geom.contains(sgeo.Point(obstacle[0])) and geom.contains(sgeo.Point(obstacle[1])) \
                and geom.contains(sgeo.Point(obstacle[2])) and geom.contains(sgeo.Point(obstacle[3])):                    
                    rect = sgeo.Polygon(tuple(obstacle))
                    # Also subtract cushioning area with width = radius
                    res -= (rect.area + rect.length * radius + 4 * radius * radius)
                    adjusted = True

    elif isinstance(hull, sgeo.collection.GeometryCollection):
        hull_pts = []
        res = hull.area
        for geom in hull.geoms:
            hull_pts.append((geom.exterior.xy[0], geom.exterior.xy[1]))
            for obstacle in obstacles:
                if geom.contains(sgeo.Point(obstacle[0])) and geom.contains(sgeo.Point(obstacle[1]))\
                and geom.contains(sgeo.Point(obstacle[2])) and geom.contains(sgeo.Point(obstacle[3])):                     
                    rect = sgeo.Polygon(tuple(obstacle))
                    # Also subtract cushioning area with width = radius
                    res -= (rect.area + rect.length * radius + 4 * radius * radius)
                    adjusted = True
    
    else:
        raise Exception(f"Unimplemented hull type: {type(hull)}")

    
    if verbose:
        print(f"[{folder} - {name}] Calculating hull: {(time.time()-start)}s")
        print(f"[{folder} - {name}] Hull area: {max(res, 0.0)}")

        start = time.time()


    if not areas:
        return max(res, 0.0), []
    
    # Reachable areas of each car is provided
    # The reachable area of all cars are present are calculated
    else:
        full_area = max(res, 0.0)
        return max(res, 0.0), [(areas[i]-full_area)/empty_area for i in range(len(obstacles))]
        
    
def STI_demo_precise(obstacles, name, missing = None, empty_area = 0, areas = [], folder = "", verbose = False):
    if verbose:
        start = time.time()
    
    if missing != None:
        missing_obstacle = obstacles[missing]
        obstacles = obstacles[:missing] + obstacles[missing+1:]
    
    timeDelta = 0.2
    totalTime = 1.6
    maxAcc = +8

    minAcc = -7
    maxSteerRight = 33
    initX, initY, initV, initYaw = 0, 0, 5, 0
    egoCar = Car(initX, initY, initV, initYaw, timeDelta)
    xStep = yStep = vStep = 1.0
    yawStep = radians(5)
    street_radius = 5.0

    # Setting radius of car
    radius = 0.9
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    a = radius * np.cos( theta )
    b = radius * np.sin( theta )
 

    if verbose:
        print(f"[{folder} - {name}] Init: {(time.time()-start)}s")
        start = time.time()


    steppingSet = {}
    steppingSet[-1] = [(initX, initY, initV, initYaw)]

    for t in range (int(np.ceil(totalTime / timeDelta))):
        steppingArray = []
        for initCondSet in steppingSet[t-1]:
            currInitX, currInitY, currInitV, currInitYaw = initCondSet
            
            # Max steering is 33 degrees
            for steer in [-maxSteerRight, 0, maxSteerRight]:
                for acc in [minAcc, maxAcc]:
                    egoCar.setStates(currInitX, currInitY, currInitV, currInitYaw)
                    egoCar.drive(acc, radians(steer))
                    
                    # Only compute if car is moving
                    if egoCar.velocity >= 0:
                        
                        
                        if egoCar.velocity >= 0 and not collided(egoCar, obstacles, xStep, yStep, radius) and -5.0 <= egoCar.y <= 5.0:
                            steppingArray.append((egoCar.x, egoCar.y, egoCar.velocity, egoCar.yaw))
                        

            steppingSet[t] = steppingArray
        
        if verbose:
            print("Current t in sec:", (t + 1) * timeDelta, len(steppingArray))
    
    
    if verbose:
        print(f"[{folder} - {name}] Sampling points: {(time.time()-start)}s")
        start = time.time()

    all_points = set()

    for timeStampSet in steppingSet.values():
        for initCondSet in timeStampSet:
            all_points.add((initCondSet[0] * xStep, initCondSet[1] * yStep))

    print("Total number of points:", len(all_points))

    res = 0.0
    hull_pts = [0, 0]
    
    alpha = 1

    all_points_np = np.array(list(all_points))
    hull  = alpha_shape(all_points_np, alpha)


    adjusted = False
    
    
    # Polygon
    if isinstance(hull, sgeo.Polygon):
        hull_pts = hull.exterior.coords.xy
        res = hull.area
        for obstacle in obstacles:
            if hull.contains(sgeo.Point(obstacle[0])) and hull.contains(sgeo.Point(obstacle[1])) \
            and hull.contains(sgeo.Point(obstacle[2])) and hull.contains(sgeo.Point(obstacle[3])):
               rect = sgeo.Polygon()
               res -= (rect.area + rect.length * radius + 4 * radius * radius)
               adjusted = True
    
    # Line
    elif isinstance(hull, sgeo.LineString):
        hull_pts = hull.convex_hull.xy

    # Multipolygon
    elif isinstance(hull, sgeo.multipolygon.MultiPolygon):
        hull_pts = []
        res = hull.area
        for geom in hull.geoms:
            hull_pts.append((geom.exterior.xy[0], geom.exterior.xy[1]))
            for obstacle in obstacles:
                if geom.contains(sgeo.Point(obstacle[0])) and geom.contains(sgeo.Point(obstacle[1])) \
                and geom.contains(sgeo.Point(obstacle[2])) and geom.contains(sgeo.Point(obstacle[3])):                    
                    rect = sgeo.Polygon(tuple(obstacle))
                    # Also subtract cushioning area with width = radius
                    res -= (rect.area + rect.length * radius + 4 * radius * radius)
                    adjusted = True

    elif isinstance(hull, sgeo.collection.GeometryCollection):
        hull_pts = []
        res = hull.area
        for geom in hull.geoms:
            hull_pts.append((geom.exterior.xy[0], geom.exterior.xy[1]))
            for obstacle in obstacles:
                if geom.contains(sgeo.Point(obstacle[0])) and geom.contains(sgeo.Point(obstacle[1]))\
                and geom.contains(sgeo.Point(obstacle[2])) and geom.contains(sgeo.Point(obstacle[3])):                     
                    rect = sgeo.Polygon(tuple(obstacle))
                    # Also subtract cushioning area with width = radius
                    res -= (rect.area + rect.length * radius + 4 * radius * radius)
                    adjusted = True
    
    else:
        raise Exception(f"Unimplemented hull type: {type(hull)}")

    
    if verbose:
        print(f"[{folder} - {name}] Calculating hull: {(time.time()-start)}s")
        print(f"[{folder} - {name}] Hull area: {max(res, 0.0)}")

        start = time.time()

    

    if not areas:
        return max(res, 0.0), []
    
    # Reachable areas of each car is provided
    # The reachable area of all cars are present are calculated
    else:
        full_area = max(res, 0.0)
        return max(res, 0.0), [(areas[i]-full_area)/empty_area for i in range(len(obstacles))]


def compute_scenarios():   
    obstacles_mapper = {}

    # Setting actor positions
    obstacles_mapper["4-actors"] = [
                    [(4.0, 1.0), (4.0, 3.0), (6, 3.0), (6, 1.0)],
                    [(6.0, -1.0), (6.0, 1.0), (8, 1.0), (8, -1.0)],
                    [(8.0, -3.0), (8.0, -1.0), (10, -1.0), (10, -3.0)],
                    [(10.0, -1.0), (10.0, 1.0), (12, 1.0), (12, -1.0)],
                    ]
    
    
    start = time.time()

    for folder_name, obstacles in obstacles_mapper.items():
        print(f"Processing {folder_name}")

        empty_area, _ = STI_demo([], "full", folder=folder_name)

        areas = []

        for i in range(len(obstacles)):
            area, _ = STI_demo(obstacles, "without#"+str(i), missing = i, folder=folder_name)
            areas.append(area)
        
        area, _ = STI_demo(obstacles, "without#"+str(2), missing = 2, folder=folder_name)


        area, risks = STI_demo(obstacles, "full-approx", empty_area=empty_area, areas = areas, folder=folder_name)

    end = time.time()
    print(f'Frame processed: {(end-start)}s, empty area = {empty_area}, full area = {area}, risks = {risks}')



if __name__ == '__main__':
    compute_scenarios()