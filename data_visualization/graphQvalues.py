import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fname = "./model_03-16_12-54-19_RESULTS.csv"
data = np.genfromtxt(fname, delimiter=',')
print ("data.shape", data.shape)

end_idx = 100000
time = data[:end_idx,0]
Q_values = data[:end_idx,1]

# Calculate moving average
n=100
def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

Q_moving_av = moving_average(Q_values,n=n)
time_moving_av = np.arange(n,end_idx+1)
#print (len(Q_moving_av))
#print (len(time_moving_av))

# Plot the graph
width_blue = 0.1
plt.plot(time,Q_values, linewidth=width_blue)
width_red = 0.3
plt.plot(time_moving_av,Q_moving_av, 'r', linewidth=width_red)
# Legend
blue_patch = mpatches.Patch(color='blue', label='Maximum instantaneous Q_value of actions')
red_patch = mpatches.Patch(color='red', label='Average maximum Q_value of actions')
plt.legend(handles=[blue_patch, red_patch], prop={'size':15})
# Axes
plt.xlabel('Timestep (frames)')
plt.ylabel('Q value')
plt.title('Predicted Q values in Shimon Hero simulator')
plt.show()

