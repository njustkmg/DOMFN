import pickle as pkl
import numpy as np
import h5py
import torch
import os

from torch.utils.data import Dataset


class MosiDataset(Dataset):
    def __init__(self,
                 vision=None,
                 text=None,
                 audio=None,
                 labels=None,
                 id=None):
        '''
        Dataset for cmu-mosi data
        :param vision_data:
        :param text_data:
        :param audio_data:
        :param labels
        '''
        super(MosiDataset, self).__init__()
        self.vision_data = vision
        self.text_data = text
        self.audio_data = audio
        self.labels = labels.squeeze()

    def __getitem__(self, index):
        inst_vision = np.max(self.vision_data[index], axis=0)
        inst_text = np.max(self.text_data[index], axis=0)
        inst_audio = np.max(self.audio_data[index], axis=0)
        inst_label = self.labels[index] + 3

        return {
            'vision_embeds': torch.tensor(inst_vision, dtype=torch.float32),
            'text_embeds': torch.tensor(inst_text, dtype=torch.float32),
            'audio_embeds': torch.tensor(inst_audio, dtype=torch.float32),
            'labels': np.long(inst_label)
        }

    def __len__(self):
        return self.vision_data.shape[0]


if __name__ == '__main__':
    print(os.getcwd())
    file = '../../data/mosi_data/mosi_data.pkl'
    data = pkl.load(open(file, 'rb'))
    print(data)
