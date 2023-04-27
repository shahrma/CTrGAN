
from __future__ import print_function

import yaml


def configs_loader(cfg_file):
    with open(cfg_file, 'r') as stream:
        data_cfgs = yaml.safe_load(stream)

    for k,v in data_cfgs['DATASET'].items():
        for idx,obj in enumerate(v):
            if obj['CENTROIDS_FILE'] is not None:
                with open(obj['CENTROIDS_FILE'], 'r') as stream:
                    centroids_cfgs = yaml.safe_load(stream)
                    data_cfgs['DATASET'][k][idx].update({'CENTROIDS':centroids_cfgs[obj['NAME']]})

    return data_cfgs

