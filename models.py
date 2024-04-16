import torch 
from torchvision.models import vgg16
from torchvision.models import resnet50

class VGG16(torch.nn.Module):
    def __init__(self, model):
        super(VGG16, self).__init__()

        self.vgg16 = model
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(in_features=1000, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x0 = self.vgg16(x)
        x1 = self.relu(x0)
        x2 = self.fc(x1)
        x3 = self.sigmoid(x2)
        return x3


class RESNET50(torch.nn.Module):
    def __init__(self, model):
        super(RESNET50, self).__init__()

        self.resnet50 = model
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.fc2 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.fc3 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.fc4 = torch.nn.Linear(in_features=1000, out_features=1)

        self.d1 = torch.nn.Dropout(p=0.2, inplace=False)
        self.d2 = torch.nn.Dropout(p=0.2, inplace=False)
        self.d3 = torch.nn.Dropout(p=0.2, inplace=False)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x0 = self.resnet50(x)
        x1 = self.relu(x0)
        x2 = self.fc1(x1)
        x3 = self.relu(x2)
        x4 = self.d1(x3)

        x5 = self.fc2(x4)
        x6 = self.relu(x5)
        x7 = self.d2(x6)

        x8 = self.fc3(x7)
        x9 = self.relu(x8)
        x10 = self.d3(x9)

        x11 = self.fc4(x10)
        x12 = self.sigmoid(x11)

        return x12
