import random
import math
import copy
import numpy as np
import time
from tkinter import *

all_points = []     # array of all points that are drawn after clustering is done
WINDOW_SIZE = 720
INTERVAL = 5000
DEVIATION = 500
NUM_OF_POINTS = 1020
colors = ['dodgerblue', 'red', 'gold', 'forestgreen', 'orange', 'midnightblue', 'darkgreen', 'darkkhaki',
          'salmon', 'deeppink', 'dimgrey', 'seagreen', 'cyan', 'saddlebrown', 'springgreen', 'violetred',
          'steelblue', 'lawngreen', 'hotpink', 'slateblue', 'maroon', 'darkviolet', 'black']


# region DRAW

# function to convert point into canvas coordinates
def coordinates(point):
    size = (INTERVAL + DEVIATION) * 4
    left = WINDOW_SIZE / 2 - 1
    right = WINDOW_SIZE / 2 + 1
    return int(point[0] / size * WINDOW_SIZE + left), int(point[1] / size * WINDOW_SIZE + left), \
           int(point[0] / size * WINDOW_SIZE + right), int(point[1] / size * WINDOW_SIZE + right)


# function to draw points on canvas
def draw_points(color_offsets, update, clear):
    master = Tk()
    master.title("Clustering Simulator 2020")
    canvas = Canvas(master, width=WINDOW_SIZE + 80, height=WINDOW_SIZE + 20, bg='whitesmoke')
    canvas.pack()
    var = IntVar()
    string_var = StringVar()
    label = Label(master, textvariable=string_var)
    label.place(x=WINDOW_SIZE + 75, y=50, anchor=E)
    button = Button(master, text="Update", command=lambda: var.set(1))
    button.place(x=WINDOW_SIZE + 75, y=20, anchor=E)
    counter = 1
    for points in all_points:
        color_offset = color_offsets.pop(0)
        if clear:
            canvas.delete("all")
        for point in points.keys():
            canvas.create_oval(coordinates(point), fill=colors[points[point] + color_offset], outline='')
        if update:
            string_var.set(f"{counter}. update")
            button.wait_variable(var)
            canvas.update()
            var.set(0)
            counter += 1
        else:
            canvas.update()
            time.sleep(0.01)

    string_var.set("end")
    master.mainloop()


# endregion


# region HELP FUNCTIONS

# calculate distance between 2 points
def euclidean_dist(point_1, point_2):
    return math.sqrt(math.pow(point_1[0] - point_2[0], 2) + math.pow(point_1[1] - point_2[1], 2))


# calculate average distance of cluster
def calculate_avg(clusters):
    averages = []
    for cluster in clusters:
        distances = []
        for point in cluster["points"]:
            distances.append(euclidean_dist(point, cluster["main_point"]))
        averages.append(sum(distances) / len(cluster["points"]))

    return averages


# summarize clustering if it was successful
def summarize(clusters):
    averages = calculate_avg(clusters)
    global_average = round(sum(averages) / len(averages), 3)

    success = True
    for avg in averages:
        if avg > DEVIATION * 5:
            success = False
            break
    if success:
        print("Clustering successful")
    else:
        print("Clustering unsuccessful")
    print(f"Average distance in clusters: {global_average}")

    return global_average


# endregion


# region CREATE POINTS

# generate first 20 points
def init():
    points = {}
    first_20 = []
    for i in range(20):
        x = random.randint(-INTERVAL, INTERVAL)
        y = random.randint(-INTERVAL, INTERVAL)

        while (x, y) in points:
            x = random.randint(-INTERVAL, INTERVAL)
            y = random.randint(-INTERVAL, INTERVAL)

        points[(x, y)] = 0
        first_20.append((x, y))
    return points, first_20


# generate points from first 20 points by adding offset to one of them
def generate_points():
    points, first_20 = init()
    counter = 20
    while counter < NUM_OF_POINTS:
        point = random.choice(first_20)
        x_offset = int(random.gauss(point[0], DEVIATION))
        y_offset = int(random.gauss(point[1], DEVIATION))
        new_point = (point[0] + x_offset, point[1] + y_offset)

        if new_point in points:
            continue

        points[new_point] = 0
        counter += 1

    return points


# endregion


# region K-MEANS

# calculate new centroid and clear array of points for each cluster
def update_centroids(clusters):
    for cluster in clusters:
        if len(cluster["points"]) == 0:
            continue
        x = 0
        y = 0
        for point in cluster["points"]:
            x += point[0]
            y += point[1]
        cluster["main_point"] = (int(x / len(cluster["points"])), int(int(y / len(cluster["points"]))))
        cluster["points"] = []
    return clusters


# calculate new medoid and clear array of points for each cluster
def update_medoids(clusters):
    for cluster in clusters:
        medoid_dist = float("inf")
        medoid = None
        for i in range(len(cluster["points"])):
            dist = 0
            for j in range(len(cluster["points"])):
                if i == j:
                    continue
                if dist > medoid_dist:
                    break
                dist += euclidean_dist(cluster["points"][i], cluster["points"][j])
            if medoid_dist is None or medoid_dist > dist:
                medoid = cluster["points"][i]
                medoid_dist = dist
        cluster["main_point"] = medoid
        cluster["points"] = []
    return clusters


def k_means(k, points, main_point, remember_all):
    global all_points
    clusters = []
    already_main_points = set()

    for i in range(k):      # create "k" clusters
        centroid = random.choice(list(points.keys()))
        while centroid in already_main_points:
            centroid = random.choice(list(points.keys()))

        points[centroid] = i
        clusters.append({"main_point": centroid, "points": []})
        already_main_points.add(centroid)

    for point in points.keys():     # fill "k" clusters
        if point in already_main_points:
            continue

        for i in range(len(clusters)):
            new_euclid_dist = euclidean_dist(point, clusters[i]["main_point"])
            old_euclid_dist = euclidean_dist(point, clusters[points[point]]["main_point"])
            if new_euclid_dist < old_euclid_dist:
                points[point] = i

        clusters[points[point]]["points"].append(point)

    if remember_all:
        all_points.append(copy.deepcopy(points))

    changed = True
    while changed:  # calculate new clusters while there are changes
        changed = False
        if main_point == "centroid":
            clusters = update_centroids(clusters)
        elif main_point == "medoid":
            clusters = update_medoids(clusters)
        for point in points.keys():
            for i in range(len(clusters)):
                new_euclid_dist = euclidean_dist(point, clusters[i]["main_point"])
                old_euclid_dist = euclidean_dist(point, clusters[points[point]]["main_point"])
                if new_euclid_dist < old_euclid_dist:
                    changed = True
                    points[point] = i
            clusters[points[point]]["points"].append(point)

        if remember_all:
            all_points.append(copy.deepcopy(points))

    if not remember_all:
        all_points.append(copy.deepcopy(points))

    return clusters


# endregion


# region DIVISIVE CLUSTERING

# do divisive clustering
def divisive(k, points):
    clusters = k_means(2, points, "centroid", False)
    color_offset = 0
    all_color_offsets = [0, 0]

    while len(clusters) < k:
        averages = calculate_avg(clusters)
        cluster = clusters[averages.index(max(averages))]
        clusters.__delitem__(averages.index(max(averages)))
        if len(cluster["points"]) >= 2:
            new_points = {}
            for point in cluster["points"]:
                new_points[point] = 0
            temp_clusters = k_means(2, new_points, "centroid", False)
            clusters.append(temp_clusters[0])
            clusters.append(temp_clusters[1])
            color_offset += 2
            if color_offset + 2 >= len(colors):
                color_offset = 0
            all_color_offsets.append(color_offset)

    return all_color_offsets, clusters


# endregion


# region AGGLOMERATIVE CLUSTERING

# fill matrix with distances
def fill_matrix(points, clusters):
    matrix = np.array([[np.uint16(0) for _ in range(len(points))] for _ in range(len(points))])
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                matrix[i][j] = np.uint16(65535)
                continue
            elif matrix[i][j] == 0:
                distance = np.uint16(euclidean_dist(clusters[i]["main_point"], clusters[j]["main_point"]))
                matrix[i][j] = distance
                matrix[j][i] = distance

    return matrix


# merge 2 closest clusters
def merge_clusters(clusters, minimal_pair, color_offset):
    global all_points
    merged_points = {}
    cluster = {"main_point": (), "points": []}
    for point in clusters[minimal_pair[0]]["points"]:
        merged_points[point] = color_offset
        cluster["points"].append(point)

    for point in clusters[minimal_pair[1]]["points"]:
        merged_points[point] = color_offset
        cluster["points"].append(point)
    x = 0
    y = 0
    for point in cluster["points"]:
        x += point[0]
        y += point[1]
    cluster["main_point"] = (int(x / len(cluster["points"])), int(int(y / len(cluster["points"]))))
    all_points.append(merged_points)
    return cluster


# delete 2 closest clusters
def delete_clusters(matrix, clusters, minimal_pair):
    matrix = np.delete(matrix, minimal_pair[0], 0)
    matrix = np.delete(matrix, minimal_pair[1], 0)
    matrix = np.delete(matrix, minimal_pair[0], 1)
    matrix = np.delete(matrix, minimal_pair[1], 1)
    clusters = np.delete(clusters, minimal_pair[0], 0)
    clusters = np.delete(clusters, minimal_pair[1], 0)
    return matrix, clusters


# update matrix and add new cluster to matrix and to array of clusters
def update_matrix(matrix, clusters, cluster):
    clusters = np.append(clusters, cluster)  # add merged cluster into cluster array and matrix
    matrix = np.append(matrix, np.array([[np.uint16(65535) for _ in range(len(matrix))]]), 0)
    matrix = np.append(matrix, np.array([[np.uint16(65535)] for _ in range(len(matrix))]), 1)

    for i in range(len(matrix) - 1):
        distance = np.uint16(euclidean_dist(clusters[-1]["main_point"], clusters[i]["main_point"]))
        matrix[-1][i] = distance
        matrix[i][-1] = distance

    return matrix, clusters


# find out closest clusters
def min_pair(matrix):
    minimal = matrix.min()
    minimal_pair = np.where(matrix == minimal)
    minimal_pair = [max(minimal_pair[0][0], minimal_pair[1][0]), min(minimal_pair[0][0], minimal_pair[1][0])]

    return minimal_pair


# do agglomerative clustering
def agglomerative(k, points):
    global all_points
    clusters = []
    color_offset = 0
    for point in points:
        cluster = {"main_point": point, "points": [point]}
        clusters.append(cluster)
        points[point] = -1

    all_points.append(copy.deepcopy(points))
    clusters = np.array(clusters)
    matrix = fill_matrix(points, clusters)  # fill matrix with distances between clusters

    while len(clusters) > k:
        minimal_pair = min_pair(matrix)  # find closest pair of clusters
        cluster = merge_clusters(clusters, minimal_pair, color_offset)  # create new cluster by merging 2 closest
        matrix, clusters = delete_clusters(matrix, clusters, minimal_pair)  # delete old clusters
        matrix, clusters = update_matrix(matrix, clusters, cluster)  # update matrix and clusters
        color_offset += 1
        if color_offset == len(colors) - 1:
            color_offset = 0

    return clusters


# endregion


# main function with user menu
def start():
    global all_points, NUM_OF_POINTS, INTERVAL, DEVIATION
    average = 0
    print("""
    ***************************************
    *      Clustering Simulator 2020      *
    ***************************************""")
    print("""
________________________________________________
| Chose which clustering method you want to use: |
|   1) K-Means with centroid                     |
|   2) K-Means with medoid                       |
|   3) Divisive clustering                       |
|   4) Agglomerative clustering                  |
________________________________________________
    """)
    option = input("Your option: ")
    NUM_OF_POINTS = int(input("Number of points: "))
    INTERVAL = int(input("Interval to generate points from: "))
    DEVIATION = int(input("Standard deviation of points: "))
    k = int(input("Enter k = "))

    points = generate_points()
    start_time = time.time()
    if option == "1":
        all_points.append(copy.deepcopy(points))
        clusters = k_means(k, points, "centroid", True)
        average = summarize(clusters)
        end_time = time.time()
        print("Time:", round((end_time - start_time)/60, 3), "min")
        draw_points([0 for _ in range(1000)], True, True)
    elif option == "2":
        all_points.append(copy.deepcopy(points))
        clusters = k_means(k, points, "medoid", True)
        average = summarize(clusters)
        end_time = time.time()
        print("Time:", round((end_time - start_time)/60, 3), "min")
        draw_points([0 for _ in range(1000)], True, True)
    elif option == "3":
        all_points.append(copy.deepcopy(points))
        color_offsets, clusters = divisive(k, points)
        average = summarize(clusters)
        end_time = time.time()
        print("Time:", round((end_time - start_time)/60, 3), "min")
        draw_points(color_offsets, True, False)
    elif option == "4":
        clusters = agglomerative(k, points)
        average = summarize(clusters)
        end_time = time.time()
        print("Time:", round((end_time - start_time)/60, 3), "min")
        draw_points([0 for _ in range(1000000)], False, False)
    return average


# function for testing
def testing():
    global all_points
    num_of_tests = 5
    for k in range(2, 21):
        average = 0
        for _ in range(num_of_tests):
            all_points = []
            average += start()
        average /= num_of_tests
        print(f"{k} - clusters. {average}")


start()
