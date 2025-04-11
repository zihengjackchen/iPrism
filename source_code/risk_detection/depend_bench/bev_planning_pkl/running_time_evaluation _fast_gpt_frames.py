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

import random

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
# save_plt: flag to save the plot
# show: flag to show the plot when running
# save_file: flag to save all risk to a file, used when generating last step
def samplingDemoApprox(obstacles, name, missing = None, empty_area = 0, areas = [], folder = "", save_plt = False, verbose = False):
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
    xStep = yStep = vStep = 0.4
    yawStep = radians(5)
    street_radius = 5.0

    # Setting radius of car
    radius = 0.8
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    a = radius * np.cos( theta )
    b = radius * np.sin( theta )
 
    if save_plt:
        plt.figure(figsize=(12, 6))
        plt.axis('scaled')
        plt.xlim(-1, 26)
        plt.ylim(-street_radius-1, street_radius)
        plt.plot(a,b, linestyle='--')

        plt.axhline(y = 1, color = 'black', linestyle = '--')
        plt.axhline(y = 3, color = 'black', linestyle = '--')
        plt.axhline(y = 5, color = 'black', linestyle = '-', linewidth = 2.0)
        plt.axhline(y = -1, color = 'black', linestyle = '--')
        plt.axhline(y = -3, color = 'black', linestyle = '--')
        plt.axhline(y = -5, color = 'black', linestyle = '-', linewidth = 2.0)

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
                    if egoCar.velocity >= 0 and not collided(egoCar, obstacles, xStep, yStep, radius) and -5.0 <= egoCar.y <= 5.0:
                        bin = (int(round(egoCar.x / xStep)), int(round(egoCar.y / yStep)), int(round(egoCar.velocity / vStep)), int(round(egoCar.yaw / yawStep)))
                        steppingHisto[bin] = 1

                        if save_plt:
                            plt.scatter(bin[0]*xStep, bin[1]*yStep, c="red", s = 0.75, alpha=0.1, zorder=-1)
            steppingSet[t] = steppingHisto
    
    if verbose:
        print(f"[{folder} - {name}] Sampling points: {(time.time()-start)}s")
        start = time.time()

    all_points = set()

    for timeStampSet in steppingSet.values():
        for initCondSet in timeStampSet:
            all_points.add((initCondSet[0] * xStep, initCondSet[1] * yStep))

    
    res = 0.0
    hull_pts = [0, 0]
    
    # alpha = 1.2
    alpha = 1.5
    # alpha = 0.7

    all_points_np = np.array(list(all_points))
    try:
        hull  = alpha_shape(all_points_np, alpha)
    except:
        hull = sgeo.Polygon()


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

    
    if save_plt:

        # Plotting
        cmap = plt.cm.RdYlGn_r

        if not isinstance(hull, sgeo.multipolygon.MultiPolygon) or isinstance(hull, sgeo.collection.GeometryCollection):
            plt.scatter(hull_pts[0], hull_pts[1], color='darkgreen')
            plt.fill(hull_pts[0], hull_pts[1] ,alpha=0.5, zorder=-2, color = "green")
        
        else:
            for geo_pts in hull_pts:
                plt.scatter(geo_pts[0], geo_pts[1], color='darkgreen')
                plt.fill(geo_pts[0], geo_pts[1] ,alpha=0.5, zorder=-2, color = "green")

        for i in range(len(obstacles)):
            coord = obstacles[i] + [obstacles[i][0]]
            xs, ys = zip(*coord) #create lists of x and y values
            if areas:
                full_area = max(res, 0.0)
                sti = (areas[i]-full_area)/empty_area
                plt.fill(xs, ys, color=cmap(sti), zorder=1)
                plt.text(xs[0], ys[0], f'{round(sti, 2)}', bbox = dict(facecolor = 'yellow', alpha = 0.5), zorder = 2)
            else:
                plt.fill(xs, ys, color="blue", zorder=1)

        if missing != None:
            coord = missing_obstacle + [missing_obstacle[0]]
            xs, ys = zip(*coord) #create lists of x and y values
            plt.fill(xs, ys, color="grey", alpha=0.5, linestyle='--', linewidth = 2.0, zorder=1)
        
        title = f'timeDelta = {timeDelta}, totalTime = {totalTime}, step = {xStep}, Alpha = {round(alpha, 2)}, Reachable Area: {round(res, 2)}'
        if adjusted:
            title += " (adjusted)"
        plt.title(title)
        
        if not os.path.exists(os.path.join('/home/sheng/projects/Kinematic-poc/', folder)):
            os.mkdir('/home/sheng/projects/Kinematic-poc/' + folder)
        
        plt.savefig(f'/home/sheng/projects/Kinematic-poc/{folder + "/" + name}', dpi=200)

        if verbose:
            print(f"[{folder} - {name}] Generating plots: {(time.time()-start)}s")

    if not areas:
        return max(res, 0.0), []
    
    # Reachable areas of each car is provided
    # The reachable area of all cars are present are calculated
    else:
        full_area = max(res, 0.0)
        return max(res, 0.0), [(areas[i]-full_area)/empty_area for i in range(len(obstacles))]
        
    

def compute_scenarios(threadpool = False):   
    obstacles_mapper = {}

    counter = 0
    
    # # 1 car
    # for x in range(1, 15, 2):
    #     for y in range(-5, 4, 2):
    #         obstacles_mapper[f"1-car-{x},{y}"] = [
    #                         [(x, y+0.2), (x, y+1.8), (x+2, y+1.8), (x+2, y+0.2)]
    #                         ]
    

    # # # 2 cars, far
    # for x in range(1, 13, 2):
    #     for y in range(-5, 4, 2):
    #         for x2 in range(1, 13, 2):
    #             for y2 in range(-5, 4, 2):
    #                 if x == x2 and y==y2:
    #                     continue
    #                 obstacles_mapper[f"2-cars-{x},{y}-{x2},{y2}"] = [
    #                                 [(x, y+0.2), (x, y+1.8), (x+2, y+1.8), (x+2, y+0.2)],
    #                                 [(x2, y2+0.2), (x2, y2+1.8), (x2+2, y2+1.8), (x2+2, y2+0.2)]
    #                                 ]
            
    # # 2 cars, squeeze
    # obstacles_mapper["2-cars-squeeze-small-r"] = [
    #                 [(6.0, 1.0), (6.0, 3.0), (9, 3.0), (9, 1.0)],
    #                 [(6.0, -1.0), (6.0, -3.0), (9, -3.0), (9, -1.0)],
    #                 ]

    # # 4 cars lining up from closest to farthest
    # obstacles_mapper["4-cars-line-small-r"] = [
    #                 [(4.0, -1.0), (4.0, 1.0), (6, 1.0), (6, -1.0)],
    #                 [(6.0, -1.0), (6.0, 1.0), (8, 1.0), (8, -1.0)],
    #                 [(8.0, -1.0), (8.0, 1.0), (10, 1.0), (10, -1.0)],
    #                 [(10.0, -1.0), (10.0, 1.0), (12, 1.0), (12, -1.0)],
    #                 ]
    
    
    # randomize cars
    for i in range(1000):
        for num_cars in range(3, 7):
            counter += 1
            obstacles_mapper[f"{num_cars}-cars-random-{i}"] = []
            for _ in range(num_cars):
                x = random.randint(1, 13)
                y = random.randint(-5, 3)
                obstacles_mapper[f"{num_cars}-cars-random-{i}"].append([(x, y+0.2), (x, y+1.8), (x+2, y+1.8), (x+2, y+0.2)])


    # # 6 cars, real-world scenario
    # obstacles_mapper["6-cars-real-world-small-r"] = [
    #                 [(2.0, 1.0), (2.0, 3.0), (5, 3.0), (5, 1.0)], 
    #                 [(6.0, 1.0), (6.0, 3.0), (9, 3.0), (9, 1.0)],
    #                 [(6.5, -1.0), (6.5, -3.0), (9.5, -3.0), (9.5, -1.0)],
    #                 [(15, -3.0), (15, -5.0), (19, -5.0), (19, -3.0)],
    #                 [(13, 2.0), (13, 4.5), (19, 4.5), (19, 2.0)],
    #                 [(20, -3.0), (20, 0), (25, 0), (25, -3.0)]
    #                 ]
    
    # # 6 cars, two lines
    # obstacles_mapper["6-cars-2-lines-small-r"] = [
    #                 [(3.0, 2.0), (3.0, 4.0), (5, 4.0), (5, 2.0)], 
    #                 [(6.0, 2.0), (6.0, 4.0), (9, 4.0), (9, 2.0)],
    #                 [(10, 2.0), (10, 4.0), (13, 4.0), (13, 2.0)],
    #                 [(3.0, -2.0), (3.0, -4.0), (5, -4.0), (5, -2.0)], 
    #                 [(6.0, -2.0), (6.0, -4.0), (9, -4.0), (9, -2.0)],
    #                 [(10, -2.0), (10, -4.0), (13, -4.0), (13, -2.0)],
    #                 ]
    
    
    start = time.time()
    count = 0
    if not threadpool:
        
        for folder_name, obstacles in obstacles_mapper.items():
            print(folder_name)
            
            count += 1
            saved_name = folder_name

            folder_name = "gpt-scenario-test/alpha=1.5"
            
            
            print(f"Processing {folder_name}")

            empty_area, _ = samplingDemoApprox([], "full", folder=folder_name, save_plt=False)

            areas = []

            for i in range(len(obstacles)):
                area, _ = samplingDemoApprox(obstacles, "without#"+str(i), missing = i, folder=folder_name, save_plt=False)
                areas.append(area)
            
            area, risks = samplingDemoApprox(obstacles, saved_name, empty_area=empty_area, areas = areas, folder=folder_name, save_plt=False)
    
        end = time.time()
        print(f'(Serial) Frames processed: {(end-start)}s, average {(end-start)/count}s')


    # EXPERIMENTAL: Concurrently calculating area diff for each car
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for folder_name, obstacles in obstacles_mapper.items():
                areas = []
                futures = []
                folder_name = "threadpool/"+folder_name

                
                futures.append(executor.submit(samplingDemoApprox, obstacles=[], name="empty", folder=folder_name))    

                for i in range(len(obstacles)):
                    futures.append(executor.submit(samplingDemoApprox, obstacles=obstacles, name="without#"+str(i), missing = i, folder=folder_name, save_plt=False))    
                for future in futures:
                    areas.append(future.result()[0])            
                # samplingDemoApprox(obstacles, name="full", full_area=areas[0], empty_area=empty_area, areas = areas[1:], folder=folder_name, save_plt=False, save_file=True)
                # futures.append(executor.submit(samplingDemoApprox, obstacles=obstacles, name="full", empty_area=areas[0], areas = areas[1:], folder=folder_name, save_plt=False, save_file=True))

            full_area, risks = samplingDemoApprox(obstacles, name="full", empty_area=areas[0], areas = areas[1:], folder=folder_name, save_plt=True)
        
            end = time.time()
            print(f'(Threadpool) Frame processed: {(end-start)}s, risks = {risks}.')

if __name__ == '__main__':
    # compute_scenarios(threadpool=True)
    compute_scenarios(threadpool=False)


