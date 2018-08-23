import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.normal_(m.weight.data,std=0.015)

    if isinstance(m,nn.Linear):
        m.weight.data.normal_(0, 0.015)
        m.bias.data.normal_(0, 0.015)
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
        # nn.init.kaiming_normal_(m.weight.data)
    #     # m.weight.data.normal_(0.0, 0.01)
    # elif classname.find('Linear') != -1:
    #     m.weight.data.normal_(0, 0.1)

class SCSF(nn.Module):
    def __init__(self,n_kernel=128):
        super(SCSF, self).__init__()
        self.n_kernel=n_kernel
        self.conv1 = nn.Conv2d(1, self.n_kernel, kernel_size=5,stride=1,padding=2)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(self.n_kernel*14*14, 10)

    def forward(self, x):
        # stride – the stride of the window. Default value is kernel_size, thus it is 2 here.
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2,stride=2))
        x = self.conv2_drop(x)
        x = x.view(-1, self.n_kernel*14*14)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)

class DCDF_Net(nn.Module):
    def __init__(self,conv1_size,conv2_size,fc1_size):
        super(DCDF_Net, self).__init__()
        self.conv1_size=conv1_size
        self.conv2_size=conv2_size
        self.fc1_size=fc1_size
        # self.conv1 = nn.Conv2d(1, self.conv1_size, kernel_size=5,stride=1,padding=2)
        # self.conv2 = nn.Conv2d(self.conv1_size, self.conv2_size, kernel_size=5,stride=1,padding=2)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(self.conv2_size*7*7, self.fc1_size)
        # self.fc2 = nn.Linear(self.fc1_size, 10)
        self.conv1=nn.Sequential(
            nn.Conv2d(1, self.conv1_size, kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(self.conv1_size),
            # stride (the stride of the window) : Default value is kernel_size
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(self.conv1_size, self.conv2_size, kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(self.conv2_size),
            # stride (the stride of the window) : Default value is kernel_size
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU()
        )
        self.fc=nn.Sequential(
            nn.Linear(self.conv2_size*7*7, self.fc1_size),
            nn.ReLU(),
            nn.Linear(self.fc1_size, 10)
        )

        self.fc1 = nn.Linear(self.conv2_size*7*7, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, 10)

    def forward(self, x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=out.reshape(out.size(0),-1)
        out=self.fc(out)
        return F.log_softmax(out, dim=1)