
from __future__ import print_function

import yaml


def configs_loader(cfg_file):
    with open(cfg_file, 'r') as stream:
        data_cfgs = yaml.safe_load(stream)

    for k,v in data_cfgs['DATASET'].items():
        if v['CENTROIDS_FILE'] is not None:
            with open(v['CENTROIDS_FILE'], 'r') as stream:
                centroids_cfgs = yaml.safe_load(stream)
                data_cfgs['DATASET'][k].update(centroids_cfgs)

    return data_cfgs

