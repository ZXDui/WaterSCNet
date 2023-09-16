import numpy as np
import os
from glob import *
import random


def get_list(root_dir, save_dir1, save_dir2):
    filelist = glob(os.path.join(root_dir, '*.hdr'))
    len_flist = len(filelist)
    print(len_flist)

    sample_num = int(len_flist * 0.8)  # 6:2:2
    ss_idx = random.sample(range(1, len_flist + 1), sample_num)

    s1 = list(range(1, len_flist + 1))
    s1_dict = {x: 1 for x in s1}
    for i in ss_idx:
        if i in s1:
            s1_dict[i] = 0
    s_out = [x for x in s1_dict if s1_dict[x] == 1]

    n_trainval = np.array(ss_idx)
    n_test = np.array(s_out)
    np.save(save_dir1, n_trainval)
    np.save(save_dir2, n_test)


if __name__ == '__main__':
    path = 'Pleace your hdr126_crop path'
    save_path1 = 'Pleace your save trainval_list path'
    save_path2 = 'Pleace your save test_list path'

    get_list(path, save_path1, save_path2)
