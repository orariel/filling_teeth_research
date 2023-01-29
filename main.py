import pyvista as pv
import numpy as np
import copy
from math import sqrt

def fix_order(fit_cr):
    fit_cr_centroid = fit_cr.mean(axis=0)
    angle_list_unsort = []
    for i in range(1, len(fit_cr)):
        delta_0 = fit_cr[i] - fit_cr_centroid
        angle_0 = np.arctan2(delta_0[1], delta_0[0]) * 180 / np.pi
        angle_list_unsort.append([angle_0, fit_cr[i][0], fit_cr[i][1], fit_cr[i][2]])
    angle_list_unsort = np.array(angle_list_unsort)
    angle_list_sort = angle_list_unsort[angle_list_unsort[:, 0].argsort(), :]
    curve_sorted = angle_list_sort[:, 1:]

    return curve_sorted

def identical_rows(arr1, arr2):

    result = []
    for i, point1 in enumerate(arr1):
        for j, point2 in enumerate(arr2):
            if np.all(point1 == point2):
                result.append(j)
                break
    return result

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)
def euclidean_distance(row1, row2):
    # calculate the Euclidean distance between two vectors
    return sqrt(((row1[0] - row2[0]) ** 2) + ((row1[1] - row2[1]) ** 2) + ((row1[2] - row2[2]) ** 2))


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append(dist)

    ind = np.argmin(distances)
    neighbors = train[int(ind)]

    return neighbors


def get_sorted_arr(arr, index):
    """sorting arr of points XYZ.
      Parameters
      ----------
      arr : np.ndarray 
          arr to sort.
      index : int 
          index to stat the sorting.
      Returns
      -------
      arr_sorted: np.ndarray
    """  #
    dataset = copy.deepcopy(arr)
    order_list = []  # this list aims to collect the order labels according to distances
    index_first = index  # first index (one of side minimum), for exemple index_first =10
    order_list.append(index_first)
    dataset_removed = copy.deepcopy(dataset)
    # this dataset will be the same of your original one but you will replace values during the process
    dataset_removed[order_list[0]] = [0, 0, 0]
    neighbors1 = get_neighbors(dataset_removed, dataset[order_list[0]], 1)

    # To find the index of the dataset for the values of the neighbors

    cond1 = np.logical_and(dataset[:, 0] == neighbors1[0], dataset[:, 1] == neighbors1[1])
    cond2 = np.logical_and(cond1, dataset[:, 2] == neighbors1[2])
    index_pos = np.where(cond2)

    # add this index in the list
    order_list.append(index_pos[0][0])
    dataset_removed[order_list[1]] = [0, 0, 0]

    # make the same thing in the following loop for
    for index_range in range(1, len(dataset)):
        dataset_removed[order_list[index_range]] = [0, 0, 0]
        neighbors2 = get_neighbors(dataset_removed, dataset[order_list[index_range]], 1)

        cond1 = np.logical_and(dataset[:, 0] == neighbors2[0], dataset[:, 1] == neighbors2[1])
        cond2 = np.logical_and(cond1, dataset[:, 2] == neighbors2[2])
        index_pos = np.where(cond2)

        if index_range == len(dataset) - 1:
            break

        order_list.append(index_pos[0][0])

    arr_list = list(dataset)
    arr_list2 = [arr_list[i] for i in order_list]

    for i in range(len(arr_list2) - 1):
        dist = euclidean_distance(arr_list2[i], arr_list2[i + 1])
        if dist > 5:
            arr_list2[i] = [0, 0, 0]
            arr_list2[i + 1] = [0, 0, 0]

    arr_sorted = np.array(arr_list2)
    arr_sorted = arr_sorted[~np.all(arr_sorted == 0, axis=1)]  # arr_list3 its you point cloud after sorting

    return arr_sorted


a = 0.5 # weight for intial normal
b = 1-a  # weight for new normal
mesh= pv.read("ball_test.stl")
all_points = np.asarray(mesh.points)
all_normals = np.asarray(mesh.point_normals)
combine_mat = np.concatenate((all_points,all_normals),axis=1)
mesh_b = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)

bound_arr =np.asarray(mesh_b.points)
bound_arr_sorted =fix_order(bound_arr)

poly = pv.PolyData(bound_arr_sorted)



#-----------------------------------------------Normals correction----------------------------------------------------------


indexes_of_bounds_after = identical_rows(bound_arr_sorted,all_points) # the indexes of the boundary points in all points array. indexes of the boundary points in all points array.
bounds_normals= all_normals[indexes_of_bounds_after]


current_vertex = 10
n= 5 # number of neighbors
e_l=[]
e_r = []
points_r =[]
points_l=[]
#-----------Caluc e_l--------------------
for i in range(current_vertex-1,current_vertex-1-n,-1):
  e_l.append(bound_arr[i]-bound_arr_sorted[current_vertex])
  points_l.append(bound_arr[i,:])
  print(current_vertex,i)
#-----------Caluc e_r-------------------
for i in range(current_vertex+1,current_vertex+n+1):
  e_r.append(bound_arr[i]-bound_arr_sorted[current_vertex])
  points_r.append(bound_arr[i,:])
  print(current_vertex,i)

points_l=np.asarray(points_l)
points_r=np.asarray(points_r)

e_l = np.array(e_l)
e_r = np.array(e_r)
e_l = normalize_vector([sum(x) for x in zip(*e_l)])
e_r = normalize_vector([sum(x) for x in zip(*e_r)])
n_e = normalize_vector(np.cross(e_l,e_r))
n_i = bounds_normals[current_vertex]
n_c = a*n_i+n_e*b
arrow_el = pv.Arrow(bound_arr[current_vertex],e_l)
arrow_er = pv.Arrow(bound_arr[current_vertex],e_r)
arrow_ne = pv.Arrow(bound_arr[current_vertex],n_e)
arrow_n = pv.Arrow(bound_arr[current_vertex],n_i)
arrow_n_c = pv.Arrow(bound_arr[current_vertex],n_c)

poly = pv.PolyData(bound_arr_sorted)
poly["My Labels"] = [f"Label {i}" for i in range(poly.n_points)]
p=pv.Plotter()
p.add_mesh(mesh)
# p.add_point_labels(poly, "My Labels", point_size=20, font_size=27)
p.add_mesh(arrow_ne,color="red")
p.add_mesh(arrow_er,color="red")
p.add_mesh(arrow_el,color="red")
p.add_mesh(arrow_n,color="blue")
p.add_mesh(arrow_n_c,color="yellow")
p.show()
