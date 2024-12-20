import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, out_feature=4):
        super(BaseModel, self).__init__()

        self.conv1 = nn.Conv2d(3,32,3,1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.conv5 = nn.Conv2d(256,512,3,1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)

        self.golbal_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        self.fc_lbp = nn.Sequential(
            nn.Linear(65536, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

        self.fc = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_feature)
        )

    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.golbal_pool(x)

        x = torch.flatten(x, 1)

        # x = self.dropout(x)

        x = self.fc(x)

        return x





