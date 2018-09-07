import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import argparse


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

if __name__ == "__main__":
   AP = argparse.ArgumentParser()
   AP.add_argument("--input",type=str,default='/home/ubuntu/Data/gelsight/input_actions.npy',help="Input Fname for actions. It is currently configured to run on AWS instances but can run anywhere with the right path")
   AP.add_argument("--output_path",type=str,default='/home/ubuntu/Data/gelsight/',help="Output path for histogrammed output. It is currently configured to run on AWS instances but can run anywhere with the right path")
   parsed = AP.parse_args()
   actions = np.load(parsed.input)
   theta, rho = cart2pol(actions[:,0],actions[:,1])
   #histogram
   hist_theta = np.histogram(theta)
   hist_rho = np.histogram(rho)
   #digitize
   theta_bin = np.digitize(theta,hist_theta[1])
   rho_bin = np.digitize(rho,hist_rho[1])
   print("Max is {} , Min is {} and Mean is {} for theta".format(theta_bin.max(),theta_bin.min(),theta_bin.mean()))
   print("Max is {}, Min is {} and mean is {} for rho".format(rho_bin.max(),rho_bin.min(),rho_bin.mean()))
   if hist_rho[0].shape != rho_bin.max():
      print("We have an extra bin we will fold the max into the last category")
      idx =np.where(rho_bin == rho_bin.max())
      rho_bin[idx] = rho_bin.max()-1
   if hist_theta[0].shape != theta_bin.max():
      print("We have an extra bin we will fold the max into the last category")
      idx =np.where(theta_bin == theta_bin.max())
      theta_bin[idx] = theta_bin.max()-1
   #MultiLabelBinarizer
   enc = MultiLabelBinarizer(classes=np.arange(1,theta_bin.max()+1))
   theta_one_hot = enc.fit_transform(theta_bin.reshape([-1,1]))
   enc = MultiLabelBinarizer(classes=np.arange(1,rho_bin.max()+1))
   rho_one_hot = enc.fit_transform(rho_bin.reshape([-1,1]))
   #saving
   np.save(parsed.output_path + '/rho_theta_actions.npy',[rho_bin,theta_bin])
   np.save(parsed.output_path + '/rho_theta_one_hot.npy',[rho_one_hot,theta_one_hot])
   np.save(parsed.output_path + '/rho_theta_hist.npy',[hist_rho,hist_theta])
