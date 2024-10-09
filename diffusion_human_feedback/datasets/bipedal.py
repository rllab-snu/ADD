import random
import numpy as np
from PIL import Image
import os


def generate_bipedal_data(
    num_control_points=8, padding=0, num_samples=100000, data_dir="bipedal_data"
):

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for data_idx in range(num_samples):

        control_points = np.zeros((num_control_points + padding, 1), dtype=np.float32)
        control_points[:num_control_points, 0] = np.random.uniform(
            size=num_control_points
        )

        np.save(data_dir + "/%07d.npy" % data_idx, control_points)

        if data_idx % 100 == 0:
            print(data_idx)


if __name__ == "__main__":

    print("Generating Data")

    generate_bipedal_data(num_samples=1000000)
