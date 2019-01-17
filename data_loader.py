import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
import json


class MvsoDataset(data.Dataset):

    def __init__(self, root, json, vocab, transform=None):
        self.root = root
        self.mvso_data = json
        self.vocab = vocab
        self.transform = transform
        self.ids = list(self.mvso_data.keys())

    def __getitem__(self,index):

        mvso_data = self.mvso_data
        vocab = self.vocab
        id = self.ids[index]
        caption = mvso_data[id]['caption']
        img_path = mvso_data[id]['image_path']
        target = mvso_data[id]['target']

        image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        caption_tensor = torch.Tensor(caption)

        return image, caption_tensor,target


    def __len__(self):
        return len(self.ids)



def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images,captions,targets = zip(*data)

    images = torch.stack(images,0)
    targets = torch.LongTensor(targets)
    lengths = [len(cap) for cap in captions]
    input_captions = torch.zeros(len(captions),50).long()
    for i, cap in enumerate(captions):

        end = lengths[i]
        #取前50个
        if end > 50:
            input_captions[i, :50] = cap[:50]
        else:
            input_captions[i,:end] = cap[:end]

    return images,input_captions,targets

def get_loader(root,json_root,vocab,transform,batch_size,shuffle,num_workers):
    with open(json_root,'rb') as f:
        json_file = json.load(f)
        f.close()
    mvso_data = MvsoDataset(root=root,json = json_file,vocab=vocab,transform=transform)

    data_loader = data.DataLoader(dataset=mvso_data,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader

'''
test

output_val = '/media/theo/data/MVSO/images/resized_val/'
val_json_root = '/home/theo/Xli/val.json'

transform = transforms.Compose([
    transforms.ToTensor()
])
with open('vocab.pkl','rb') as f:
    vocab = pickle.load(f)
    f.close()
val = get_loader(output_val,val_json_root,vocab,transform,batch_size=8,shuffle=True,num_workers=12)
for ii,(imgs,captions,targets) in enumerate(val):
    if ii > 1:
        break
    print(targets)
    print(targets.size())
'''