import numpy
import copy
import pyvista as pv
import numpy as np

n=3
crev= np.loadtxt("c.txt")
current_index=0

e_r = []
e_l = []
k=1
crev = copy.deepcopy(crev)
current_index = copy.deepcopy(current_index)
offset=int(crev.shape[0])

for current_index in range(5,10):
    e_r = []
    e_l = []
    for i in range(current_index + 1, current_index + n + 1):

        if (i > int(crev.shape[0])-1):
            t = i - int(crev.shape[0])
            e_r.append(crev[t] - crev[current_index])
            print(t, current_index)

        else:
            e_r.append(crev[i] - crev[current_index])
            print(i, current_index)

    for i in range(current_index - 1, current_index - 1 - n, -1):
        #
        #  e_previous.append(crev[i] - crev[current_index])
        e_l.append(crev[i] - crev[current_index])
        print(i,current_index)

        #
        # else:
        #     e_r.append(crev[i] - crev[current_index])
        #     print(i, current_index)
    e_l = np.array(e_l)
    e_r = np.array(e_r)
    e_mid = e_l + e_r
    vertex_current = np.asarray(crev[current_index, :]).reshape(1, 3)
    vertex_new = vertex_current + l_avg * e_mid
    new_points.append(vertex_new)

p=pv.Plotter()

#p.add_mesh(arrow_ne,color="red")
# p.add_mesh(arrow_er,color="red")
p=pv.Plotter()
p.add_mesh(pv.PolyData(new_points),color="black")
p.show()