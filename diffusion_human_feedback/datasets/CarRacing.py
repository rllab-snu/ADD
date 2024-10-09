import random
import numpy as np
from PIL import Image
import os 

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_random_points(n=12, mindst=None, rec=0, **kw):
    """Create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or 0.7/n
    np_random = kw.get('np_random', np.random)
    a = np_random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a
    else:
        return get_random_points(n=n, 
            mindst=mindst, rec=rec+1, np_random=np_random)


def generate_CarRacing_data(num_control_points = 12, channels=2, padding=0, num_samples=100000, data_dir="CarRacing_2D"):
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for data_idx in range(num_samples):
        
        # control_points = get_random_points(n=num_control_points)
        # random_padding = np.random.rand(padding, 2)
        # control_points = np.concatenate((control_points, random_padding), axis=0)
        # control_points = control_points.astype(np.float32)
        control_points = np.random.rand(num_control_points+padding,channels)
        control_points = control_points.astype(np.float32)
        
        np.save(data_dir + "/%07d.npy"%data_idx, control_points)
        
        if(data_idx % 100 == 0):
            print(data_idx)

def generate_CarRacing_data_Poisson(num_control_points = 12, channels=2, padding=0, num_samples=100000, data_dir="CarRacing_2D_Poisson"):
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    for data_idx in range(num_samples):
        
        control_points = np.random.rand(num_control_points + padding, channels)
        control_points = control_points.astype(np.float32)
        # np.zeros((num_control_points+padding,channels), dtype=np.float32)
        # control_points[:num_control_points, 0] = np.random.uniform(size=12)
        # control_points[:num_control_points, 1] = np.random.uniform(size=12)
        
        np.save(data_dir + "/%07d.npy"%data_idx, control_points)
        
        if(data_idx % 100 == 0):
            print(data_idx)


if __name__ == "__main__":
    
    print("Generating Data")
    
    generate_CarRacing_data(num_control_points=12, num_samples=10000000)