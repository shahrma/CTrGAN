import os

def makedirs(d):
    p,f,e = fileparts(d)
    os.makedirs(p,exist_ok=True)

def fileparts(filename):
    p = os.path.dirname(filename)
    f = os.path.basename(filename)
    f, e = os.path.splitext(f)
    return p, f, e


def get_files_list_(dir_name):
    # create a list of file and sub directories
    # names in the given directory
    files_list = os.listdir(dir_name)

    all_files = list()
    # Iterate over all the entries
    for entry in files_list:
        # Create full path
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            all_files = all_files + get_files_list_(full_path)
        else:
            all_files.append(full_path)
    return all_files

def get_dirs_list_(dir_name):
    # create a list of file and sub directories
    # names in the given directory
    files_list = os.listdir(dir_name)

    all_dirs = list()
    # Iterate over all the entries
    for entry in files_list:
        # Create full path
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            all_dirs = all_dirs + get_dirs_list_(full_path)
        else:
            all_dirs.append(dir_name)

    return all_dirs



def get_files_list(dir_name, types):
    fileslist = get_files_list_(dir_name)
    fileslist_ = [f for f in fileslist if os.path.isfile(os.path.join('.', f)) and f.endswith(types)]
    return fileslist_


def unique(src_list):
    unique_list = list()
    for item in src_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list
import re

import difflib as dl

def get_dirs_list(dir_name,return_diff = False):
    dirs = dir_name.split('.*')

    if len(dirs) == 1 :
        dirslist = get_dirs_list_(dirs[0])
        dirslist_ = unique(dirslist)
    else :
        dirslist = get_dirs_list_(dirs[0])
        dirslist_ = unique(dirslist)
        x = [re.match(dir_name,f) for f in dirslist_]
        dirslist_ = [f[0] for f in x if f is not None]

    y = list()
    for item in dirslist_ :
        for sp in dirs :
            item = item.replace(sp,'')
        y.append(item)

    if return_diff :
        return dirslist_,y
    else :
        return dirslist_

import json
class VariablesLogger():
    def __init__(self,logpath):
        os.makedirs(logpath,exist_ok=True)
        self.logpath = os.path.join(logpath,'vars_log.json')

    def validations(self,param,value) :
        if not os.path.isfile(self.logpath) :
            with open(self.logpath, 'w') as json_file:
                data = dict()
                data[param] = value
                json.dump(data, json_file)

    def load_param(self,param,default_value = 0):
        self.validations(param,default_value)
        with open(self.logpath, 'r') as json_file:
            data = json.load(json_file)

        if param in data.keys() :
            val = data[param]
        else :
            self.save_param(param,default_value)
            val = default_value

        return val

    def save_param(self,param,value):
        self.validations(param, value)
        with open(self.logpath, 'r') as json_file:
            data = json.load(json_file)

        with open(self.logpath, 'w') as json_file:
            data[param] = value
            json.dump(data, json_file)

