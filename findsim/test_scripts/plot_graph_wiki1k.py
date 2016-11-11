import numpy as np
import scipy as sp
#%matplotlib inline
import matplotlib.pyplot as plt # side-stepping mpl backend

fig = plt.figure()
fig.suptitle('Results for Wiki1k.csr, Rows = 1000', fontsize=14, fontname="Times New Roman",fontweight="bold")

ax1 = fig.add_subplot(2,2,1)

dyn_x_axis = [0.5,
0.7,
0.9]

#----- For k = 10 -----------------
ij_y_axis = [0.0976,
0.1226333333,
0.09846666667]

dyn_y_axis = [0.0418,
0.04386666667,
0.041]

#----- For k = 50 -----------------
ij_50_y_axis = [0.09376666667,
0.0945,
0.0947]

dyn_50_y_axis = [0.0441,
0.04063333333,
0.0427]

#----- For k = 100 -----------------
ij_100_y_axis = [0.0933,
0.09733333333,
0.0958]

dyn_100_y_axis = [0.04093333333,
0.04426666667,
0.04616666667]


#ax1.set_title('k = 10')
ax1.set_xlabel('Eps')
ax1.set_ylabel('Search time (sec)')
ax1.text(0.9, 0.8,'k = 10', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
ax1.plot(dyn_x_axis, dyn_y_axis, label='dynamic_index', marker='o')
ax1.plot(dyn_x_axis, ij_y_axis, color='r', label='index_join', marker='o')
ax1.grid('on')


ax2 = fig.add_subplot(2,2,2)
#ax2.set_title('k = 10')
ax2.set_xlabel('Eps')
ax2.set_ylabel('Search time (sec)')
ax2.text(0.9, 0.8,'k = 50', ha='center', va='center', transform=ax2.transAxes, fontsize=14,)
ax2.plot(dyn_x_axis, dyn_50_y_axis, label='dynamic_index', marker='o')
ax2.plot(dyn_x_axis, ij_50_y_axis, color='r', label='index_join', marker='o')
ax2.grid('on')


ax0 = fig.add_subplot(2,2,3)
#ax0.set_title('k = 10')
ax0.set_xlabel('Eps')
ax0.set_ylabel('Search time (sec)')
ax0.text(0.9, 0.8,'k = 100', ha='center', va='center', transform=ax0.transAxes, fontsize=14)
ax0.plot(dyn_x_axis, dyn_100_y_axis, label='dynamic_index', marker='o')
ax0.plot(dyn_x_axis, ij_100_y_axis, color='r', label='index_join', marker='o')
ax0.grid('on')

# Put a legend below current axis
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
          fancybox=True, shadow=True, ncol=2)

fig.patch.set_facecolor('white')
plt.show()
