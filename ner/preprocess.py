import os
from .util_preprocess import read_examples_data


def dataner_preprocess(data_dir, file_name):
    print(data_dir)

    path = os.path.join(data_dir, file_name)

    # Read of examples
    examples = read_examples_data(path)
    return examples
