import pyvista as pv
import numpy as np

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


current_vertex = 5
n= 3 # number of neighbors
e_l=[]
e_r = []
#-----------Caluc e_l--------------------
for i in range(current_vertex-1,current_vertex-1-n,-1):
  e_l.append(bound_arr[i]-bound_arr_sorted[current_vertex])
#-----------Caluc e_r-------------------
for i in range(current_vertex+1,current_vertex+n+1):
  e_r.append(bound_arr[i]-bound_arr_sorted[current_vertex])
  print(current_vertex,i)

e_l = np.array(e_l)
e_r = np.array(e_r)
e_l = normalize_vector([sum(x) for x in zip(*e_l)])
e_r = normalize_vector([sum(x) for x in zip(*e_r)])
n_e = normalize_vector(np.cross(e_l,e_r))
n_i = bounds_normals[0]
n_c = a*n_i+n_e*b
arrow_el = pv.Arrow(bound_arr[current_vertex],e_l)
arrow_er = pv.Arrow(bound_arr[current_vertex],e_r)
arrow_ne = pv.Arrow(bound_arr[current_vertex],n_e)
arrow_n = pv.Arrow(bound_arr[current_vertex],n_i)
arrow_n_c = pv.Arrow(bound_arr[current_vertex],n_c)


poly["My Labels"] = [f"Label {i}" for i in range(poly.n_points)]
p=pv.Plotter()
p.add_mesh(mesh)
p.add_point_labels(poly, "My Labels", point_size=20, font_size=27)
p.add_mesh(arrow_ne,color="red")
p.add_mesh(arrow_er,color="red")
p.add_mesh(arrow_el,color="red")
p.add_mesh(arrow_n,color="blue")
p.add_mesh(arrow_n_c,color="yellow")
p.show()
