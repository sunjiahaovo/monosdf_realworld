import json
import os
import numpy as np

root_dir = "/remote-home/ums_sunjiahao/monosdf_real/data/L435/scan5"
with open(os.path.join(root_dir, f"transforms_train.json"), 'r') as f:
    meta = json.load(f)
# print(meta)

for i in range(100):
    
    frame = meta['frames'][i]
    path_str = frame['file_path']
    print(i,path_str)
    path_arr = path_str.split("/")
    print(path_arr)
    index = int(path_arr[4].split("_")[0])
    print(index)
    T = np.array(frame['transform_matrix'])
    print(T)
    # print(frame)
    np.savetxt(os.path.join(root_dir,"l435_out_384",str(index)+".txt"),T)