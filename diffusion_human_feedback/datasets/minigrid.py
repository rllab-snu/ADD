import random
import numpy as np
from PIL import Image
import os 

num_walls = 60
data_dir = "minigrid_" + str(num_walls) + "_uniform"

def generate_minigrid_data(grid_size=13, target_img_size=16, num_walls=60, num_samples=10000, filename='minigrid_train', chunk_size=32, data_dir="minigrid_50"):
    
    base = list(range(grid_size ** 2))
    num_blocks = []
    for data_idx in range(num_samples):
        
        wall_idxs = random.choices(base, k=random.randint(0, num_walls-1))
        start_and_end = random.sample(base, 2)
        start_idx, end_idx = start_and_end[0], start_and_end[1]
        start_direction = random.randrange(4)          
        
        # wall on R channel, start on G channel, end on B channel
        # if start or goal is placed on the wall, remove the wall and place
        # start and goal never placed on the same posiiton since we have used random.sample
        grid_env = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        for wall_idx in wall_idxs:
            grid_env[wall_idx // grid_size, wall_idx % grid_size, 0] = 255
            
        grid_env[start_idx // grid_size, start_idx % grid_size, 0] = 0
        grid_env[start_idx // grid_size, start_idx % grid_size, 1] = 255
    
        grid_env[end_idx // grid_size, end_idx % grid_size, 0] = 0
        grid_env[end_idx // grid_size, end_idx % grid_size, 2] = 255
                        
        # use padding to use it as a sample of the diffusion model
        padded_img = np.pad(grid_env, ((1,2),(1,2), (0,0)), 'constant', constant_values=0)
        padded_img[0,:,0] = 255
        padded_img[:,0,0] = 255
        padded_img[-2:,:,0] = 255
        padded_img[:,-2:,0] = 255
        
        # indicate agent initial direction after padding
        x_direction = 0
        y_direction = 0
        if start_direction == 0:
            x_direction = 1
        elif start_direction == 1:
            x_direction = -1
        elif start_direction == 2:
            y_direction = 1
        elif start_direction == 3:
            y_direction = -1
        else:
            raise NotImplementedError
        
        start_idx_x, start_idx_y = np.where(padded_img[:,:,1] == 255)
        padded_img[start_idx_x + x_direction, start_idx_y + y_direction, 1] = 128         
        
        im = Image.fromarray(padded_img)
        im.save(data_dir + "/%07d"%data_idx + ".png")
        num_blocks.append(np.sum(grid_env[:,:,0] == 255))
        if(data_idx % 1000 == 0):
            print(sum(num_blocks) / len(num_blocks))

    # flat_str = lambda lst: " ".join([str(l) for l in lst])
    # s = ""
    # for seq in seqs:
    #     s += (flat_str(seq) + "\n")

    # with open(f"{filename}_{num_samples}_{chunk_size}.txt", "w+") as f:
    #     f.write(s)



if __name__ == "__main__":
    
    print("Generating Data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    generate_minigrid_data(num_walls=num_walls, num_samples=10000000, data_dir = data_dir)