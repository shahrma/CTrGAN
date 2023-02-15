
import os
import json
import numpy as np
import pandas as pd


import argparse
from common.fileutils import get_files_list,fileparts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src_root', type=str, default='..//../../workspace/RecycleGAN/', help='src images path.')
    args = parser.parse_args()

    #args.src_root = '..//../../workspace/RecycleGAN/'
    src_root = args.src_root

    flist = get_files_list(args.src_root,'opt.json')

    dflist = pd.DataFrame()
    for filename in flist :
        df = pd.read_json(filename)
        p,f,e = fileparts(filename)
        df['EXP'] = p
        dflist = pd.concat([dflist,df], axis=0, ignore_index=True)

    dflist.to_csv(os.path.join(args.src_root,'experiments.csv'))
