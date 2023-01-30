import pyvista as pv
import numpy as np
import copy
from math import sqrt
from scipy import interpolate
import open3d as o3d

def get_curve_interpolation_for_real_boundary(tooth, tooth_feature_edges, number_of_points, fit_factor):
    """
              Creating a curve interpolation to make the boundary smoother.
              Parameters
              ----------
              tooth :  mesh (pv.PolyData)
                  Mesh file of the tooth. 
              tooth_feature_edges :  pv.PolyData
                  The none minfold triangles of the real tooth.
              numbers_of_points :  int
                  Root vector normalize.
              fit_factor : int 
                  Close to 0 make the fit better (0- on the real points)
              Returns
              -------
              fit_cr2 = intepolated curve (Nx3)
       """  #
    data_b = tooth_feature_edges
    data_b_2 = data_b.connectivity()  # to detect if there is noise
    arr_index = np.asarray(data_b_2.active_scalars)
    arr_b = np.asarray(data_b_2.points)
    # arr_b=arr_b[:, 2]-offset

    arr_b_clean = arr_b
    if arr_index.max() > 0:  # remove noise
        x = arr_index
        unique, counts = np.unique(x, return_counts=True)
        a = np.asarray((unique, counts)).T
        a = a[a[:, -1].argsort()]
        num = a[-1, 0]
        points_to_remove = []
        t = 0
        for i in range(arr_b.shape[0]):
            if arr_index[i] != num:
                t += 1
                points_to_remove.append(i)
        arr_b_clean = arr_b[0:-t]
    arr_b_1 = arr_b_clean
    arr_b_1 = get_sorted_arr(arr_b_1, 0)

    data = arr_b_1.transpose()  # np.arr points boundary after sorting
    # now we get all the knots and info about the interpolated spline
    tck, u = interpolate.splprep(data, s=1, per=True)  # s- lower it fit better 5
    new = interpolate.splev(np.linspace(0, 1, number_of_points), tck, der=0)
    fit_cr2 = np.concatenate((new[0].reshape(len(new[0]), 1),
                              new[1].reshape(len(new[0]), 1),
                              new[2].reshape(len(new[0]), 1)), axis=1)

    return fit_cr2

def get_clean_feature_edges(feature_edges):
    data_b = feature_edges
    data_b_2 = data_b.connectivity()
    arr_index = np.asarray(data_b_2.active_scalars)
    arr_b = np.asarray(data_b_2.points)
    # arr_b=arr_b[:, 2]-offset

    arr_b_clean = arr_b
    if arr_index.max() > 0:
        x = arr_index
        unique, counts = np.unique(x, return_counts=True)
        a = np.asarray((unique, counts)).T
        a = a[a[:, -1].argsort()]
        num = a[-1, 0]
        points_to_remove = []
        t = 0
        for i in range(arr_b.shape[0]):
            if arr_index[i] != num:
                t += 1
                points_to_remove.append(i)
        arr_b_clean = arr_b[0:-t]
    arr_b_1 = arr_b_clean
    arr_b_1 = get_sorted_arr(arr_b_1, 0)

    return arr_b_1

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

def get_er_el(crev, current_index,n):
    e_r=[]
    e_l =[]
    crev = copy.deepcopy(crev)
    current_index= copy.deepcopy(current_index)
    for i in range(current_index - 1, current_index - 1 - n, -1):
        #
          #  e_previous.append(crev[i] - crev[current_index])
        e_l.append(crev[i] - crev[current_index])

    # -----------Caluc e_r-------------------
    for i in range(current_index + 1, current_index + n + 1):

        if (i >= int(crev.shape[0])):
            t= i-int(crev.shape[0])
            e_r.append(crev[t] - crev[current_index])
           # e_next.append(crev[i] - crev[current_index])
            print(t, current_index)
        else:
            e_r.append(crev[i] - crev[current_index])

    e_l = np.array(e_l)
    e_r = np.array(e_r)
    return e_l, e_r


def min_distance(points):
    distances = []
    distances.append(np.linalg.norm(points[0]-points[1]))
    return distances
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append(dist)

    ind = np.argmin(distances)
    neighbors = train[int(ind)]

    return neighbors

def rot_matrix(n, theta):
    n = np.asarray(n)
    n = n / np.linalg.norm(n)
    skew = np.array([[0, -n[0,2], n[0,1]], [n[0,2], 0, -n[0,0]], [-n[0,1], n[0,0], 0]])
    I = np.eye(3)
    R = np.cos(theta) * I + (1 - np.cos(theta)) * np.outer(n, n) + np.sin(theta) * skew
    return R



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
w_1 = 0.9  # params for weight function
w_2 = 1-w_1  # params for weight function
tooth= pv.read("ball_test.stl")
tooth_b = tooth.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
faces = tooth.faces
vertex_b = get_clean_feature_edges(tooth_b)
num_of_points = int(vertex_b.shape[0])
crev = get_curve_interpolation_for_real_boundary(tooth, tooth_b, int(vertex_b.shape[0]), 1)
np.savetxt("c.txt",crev)
vertex_orignal = np.asarray(tooth.points)
list_of_index = []

# <--------------------------------Modify the bound for smoother curve--------------------------------------------------->

for k in range(vertex_orignal.shape[0]):
    for j in range(vertex_b.shape[0]):
        point_a = vertex_b[j, :]
        point_b = vertex_orignal[k, :]
        if (point_a[0] == point_b[0] and point_a[1] == point_b[1] and point_a[2] == point_b[2]):
            vertex_orignal[k, :] = crev[j, :]
            list_of_index.append([k, j])
#

poly = pv.PolyData(crev)
poly["My Labels"] = [f"Label {i}" for i in range(poly.n_points)]
p = pv.Plotter()

p.add_point_labels(poly, "My Labels", point_size=20, font_size=27)
# p.add_mesh(crev,color="red")
p.add_mesh(crev[0,:],color="pink")
p.add_mesh(tooth)
# p.show()
# tooth.plot_normals()
all_points = np.asarray(tooth.points)
all_normals = np.asarray(tooth.point_normals)



#
#
#
# #-----------------------------------------------Normals correction----------------------------------------------------------
#

#indexes_of_bounds_after = identical_rows(crev,all_points) # the indexes of the boundary points in all points array. indexes of the boundary points in all points array.
#bounds_normals= all_normals[indexes_of_bounds_after]
#
#
current_index = 0
vertex_current = np.asarray(crev[current_index,:]).reshape(1,3)
n= 5 # number of neighbors

points_r =[]
points_l=[]
e_next = []
e_previous = []
new_points=[]
#-----------Caluc e_l--------------------
for i in range(crev.shape[0]-1):
    e_l, e_r = get_er_el(crev, i,n)


# #----------------------------------------------Calc curvature ---------------------------------------------------------
    l_avg= min_distance(crev)
    e_previous = np.array(e_previous)
    e_next = np.array(e_next)
    e_l = normalize_vector([sum(x) for x in zip(*e_l)])
    e_r = normalize_vector([sum(x) for x in zip(*e_r)])
    n_e = normalize_vector(np.cross(e_l,e_r))
    #n_i = bounds_normals[current_vertex]
    n_c = n_e
    arrow_el = pv.Arrow(crev[current_index], e_l)
    arrow_er = pv.Arrow(crev[current_index], e_r)
    arrow_ne = pv.Arrow(crev[current_index], n_e)
# K_=(np.transpose(n_c)*e_previous)/(np.linalg.norm(e_previous)**2)+(np.transpose(n_c)*e_next)/(np.linalg.norm(e_next)**2)  # Taubin curvature
# l_avg = 0.5*((np.linalg.norm(e_previous))+(np.linalg.norm(e_next))) # this is the avg edge of the current vertex
# p_1  = 2 * np.transpose(n_c)
# #
# e_initial = l_avg*(e_previous+e_next)
# n_s =(np.cross(n_c,e_initial))/(np.linalg.norm(np.cross(n_c,e_initial))) # To avoid degenerated triangles when creating new filling triangles, the optimal edge ei,o is restricted on the plane Ps whose normal is
# plane_p = pv.Plane(center=crev[current_index], direction=n_s)
# A_1 = w_1*l_avg*K_
# A_2=w_2*((np.transpose(n_c)*e_initial)/(np.linalg.norm(e_initial))**2)
# A=A_1+A_2
#
# teta =np.rad2deg(np.arccos(np.linalg.norm(A))) # need to check if allways in the range [-1,1]
# R_s= rot_matrix(n_s,teta)
# e_optimal =l_avg*np.dot(R_s, n_c.reshape(3, 1))
# e_optimal= np.transpose(e_optimal)
#
# vertex_new =e_initial- vertex_current

#
    e_mid = e_l+e_r
    arrow_e_mid = pv.Arrow(crev[current_index],e_mid)
    vertex_new = vertex_current+l_avg*e_mid
    new_points.append(vertex_new)
new_points= np.concatenate(new_points)
p=pv.Plotter()
p.add_mesh(tooth)
p.add_mesh(poly)
#p.add_mesh(arrow_ne,color="red")
# p.add_mesh(arrow_er,color="red")
p.add_mesh(pv.PolyData(new_points),color="black")
# p.add_mesh(arrow_el,color="red")
# p.add_mesh(arrow_e_mid,color="blue")
# p.add_mesh(arrow_n_c,color="yellow")
p.show()
