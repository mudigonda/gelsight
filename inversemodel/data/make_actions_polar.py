import numpy as np


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

if __name__ == "__main__":
   actions = np.load('/home/ubuntu/Data/gelsight/input_actions.npy')
   theta, rho = cart2pol(actions[0],actions[1])
   #histogram
   hist0 = np.histogram(theta)
   hist1 = np.histogram(rho)
   #digitize
   theta_bin = np.digitize(theta,hist0[1])
   rho_bin = np.digitize(rho,hist1[1])
   print("Max is {} , Min is {} and Mean is {} for theta".format(theta_bin.max(),theta_bin.min(),theta_bin.mean()))
   print("Max is {}, Min is {} and mean is {} for rho".format(rho_bin.max(),rho_bin.min(),rho_bin.mean()))
   #saving
   np.save('/home/ubuntu/Data/gelsight/rho_theta_actions.npy',[rho_bin,theta_bin])
