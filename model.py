import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ImCNN(nn.Module):
    def __init__(self):
        super(ImCNN,self).__init__()
        alexnet = models.alexnet()
        modules = list(alexnet.children())[:-1]
        self.alexnet = nn.Sequential(*modules)


    def forward(self, images):
        features = self.alexnet(images)
        features = features.reshape(features.size(0), -1)

        return features


class CapCNN(nn.Module):
    def __init__(self,embed_size,  vocab_size):
        super(CapCNN,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.conv1 = nn.Conv2d(1, 1, [2, 2])
        self.conv2 = nn.Conv2d(1, 1, [3, 3])
        self.conv3 = nn.Conv2d(1, 1, [5, 5])


    def conv_and_pool(self, x, conv):
        x = conv(x)
        x1 = F.max_pool2d(x, [3,3]).reshape(x.size(0), -1)
        x2 = F.avg_pool2d(x, [3,3]).reshape(x.size(0), -1)

        return torch.cat([x1,x2], 1)

    def forward(self, captions):
        x = self.embed(captions)
        x = x.unsqueeze(1)
        x_1 = self.conv_and_pool(x, self.conv1)
        x_2 = self.conv_and_pool(x, self.conv2)
        x_3 = self.conv_and_pool(x, self.conv3)

        results = torch.cat([x_1, x_2, x_3],1)
        return results


class ClassiModel(nn.Module):
    def __init__(self, feature_size, out_size):
        super(ClassiModel,self).__init__()

        self.linear = nn.Linear(feature_size, out_size)

    def forward(self, im_features, cap_features):
        x = torch.cat([im_features,cap_features],1)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)

        return x







