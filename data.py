import os

from torch.utils.data import Dataset

WORK_DIR = os.getcwd()
TRAIN_PATH = os.path.join(WORK_DIR, "data", "train")
TEST_PATH = os.path.join(WORK_DIR, "data", "test")


label2id = {label: id for id, label in enumerate(os.listdir(TRAIN_PATH))}

id2label = {id: label for label, id in label2id.items()}


def read_file(dir, label, file_name):
    text_path = os.path.join(dir, label, file_name)
    with open(text_path, "r", encoding="utf-16") as file:
        text = file.read()
    return text, label2id[label]


def read_dir(dir):
    data = []
    for label in os.listdir(dir):
        label_path = os.path.join(dir, label)
        for file_name in os.listdir(label_path):
            data.append(read_file(dir, label, file_name))
    return data


class VNTCDataset(Dataset):
    def __init__(self, train=True):
        self.data = read_dir(TRAIN_PATH if train else TEST_PATH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
