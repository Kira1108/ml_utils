import os
from dataclasses import dataclass
from shutil import copyfile
import random
import numpy as np

@dataclass
class RawDataDirectories:
    root_path:str
    categories:list
    paths:list

    @classmethod
    def from_source_path(cls, root_path):
        # list all classes
        categories = os.listdir(root_path)

        # find all classes paths
        paths = [os.path.join(root_path, c) for c in categories]
        return cls(root_path, categories, paths)

    def shuffled_file_paths(self,category, verbose = True):
        """collected shuffled file paths for given category"""
        if verbose:
            print("Shuffling dataset....")
        idx = self.categories.index(category)
        source = self.paths[idx]

        files = []
        for filename in os.listdir(source):
            file = os.path.join(source, filename)
            if os.path.getsize(file) > 0:
                files.append(file)
            else:
                if verbose:
                    print(filename + " is zero length, so ignoring.")
        if verbose:
            print("Shuffling dataset Done")
        return random.sample(files, len(files))

    def summarize(self):
        print("*"*10, "Summary", "*"*10)
        for category in self.categories:
            print(f"Category: {category}, files: {len(self.shuffled_file_paths(category, verbose = False))}")
        print("*"*30)

    def train_test_split(self, split_sizes:list = None, split_names:list = None):
        print("Spliting Dataset....")
        if split_sizes is None:
            split_sizes = [0.8, 0.1, 0.1]

        if split_names is None:
            if len(split_sizes) == 2:
                split_names = ['training','testing']
            elif len(split_sizes) == 3:
                split_names = ['training','testing','validation']
            else:
                split_names = [f'split{i + 1}' for  i in range(len(split_sizes))]

        split_sizes.insert(0,0)

        split_dataset = {}

        for category in self.categories:
            print("working on category: ", category)
            files = self.shuffled_file_paths(category)

            split_points = (len(files) * np.array(split_sizes)).astype(int).cumsum().tolist()

            splits = {split_names[i]: files[split_points[i]: split_points[i+1]] 
                            for i in range(len(split_points) - 1)}
            
            split_dataset[category] = splits
        print("Splitting Dataset Down")
        return split_dataset

def create_target_directory(root_path:str, classes:list, splits:list = None):
    """
    root_path: root directory of data path
    classes: list of class names for classification
    splits: how to split the dataset like - /training/testing/validation/...
    """
    try:
        for split in splits:
            for c in classes:
                p = os.path.join(root_path, split, c)
                if not os.path.exists(p):
                    os.makedirs(p)
                else:
                    print(f"Directory {p} already exists.")
    except OSError as e:
        print("failed to create data directory for ML.")
        raise(e)
    print("Target directory tree created")

def copy_file_to_target(split_result, target_path):
    print("Copying files....")
    for category, data in split_result.items():
        print(f"Copying file for category: {category}")
        for split_name, split_data in data.items():
            print(f"Split set: {split_name}")
            for source_file_name in split_data:
                basefilename = os.path.basename(source_file_name)
                target_file_path = os.path.join(target_path, split_name, category, basefilename)
                if not os.path.exists(target_file_path):
                    copyfile(source_file_name, target_file_path)
    print("Copying files Done.")

def prepare_cv_dataset(source_path, target_path, splits, split_sizes):
    raw_data = RawDataDirectories.from_source_path(source_path)
    raw_data.summarize()
    result = raw_data.train_test_split(split_sizes = split_sizes)
    create_target_directory(target_path, raw_data.categories, splits)
    copy_file_to_target(result, target_path)
    print("Finish prepare ml image classification dataset")