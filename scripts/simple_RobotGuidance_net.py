import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser


import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from yaml import load


# HYPER PARAM
BATCH_SIZE = 8  #トレーニング中にネットワークに一度に入力されるサンプルの数
MAX_DATA = 10000    #データセット内の最大データ数を制限するためのパラメータ


class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    # <Network CNN 3 + FC 2>
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)
    # <Weight set>
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        # torch.nn.init.kaiming_normal_(self.fc5.weight)
        #self.maxpool = nn.MaxPool2d(2,2)
        #self.batch = nn.BatchNorm2d(0.2)
        self.flatten = nn.Flatten()
    # <CNN layer>
        self.cnn_layer = nn.Sequential( #畳み込み層とReLU活性化関数を順番に適用するシーケンシャルな層
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            # self.maxpool,
            self.flatten
        )
    # <FC layer (output)>
        self.fc_layer = nn.Sequential(  #全結合層とReLU活性化関数を順番に適用するシーケンシャルな層
            self.fc4,
            self.relu,
            self.fc5,

        )

    # <forward layer>
    def forward(self, x):
        x1 = self.cnn_layer(x)  # CNNレイヤーにデータを入力し、特徴マップを生成
        x2 = self.fc_layer(x1)  # 全結合レイヤーに特徴マップを入力し、最終的な出力を生成
        return x2   # ネットワークの出力を返す


class deep_learning:
    def __init__(self, n_channel=3, n_action=3):    #n_channel=3はカラー画像，n_actionは出力のアクション数
        # <tensor device choiece>
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(n_channel, n_action)
        self.net.to(self.device)
        print(self.device)  #'cuda'or'cpu'が出力される
        self.optimizer = optim.Adam(    #モデルの重みを最適化するための最適化アルゴリズムを指定
            self.net.parameters(), eps=1e-2, weight_decay=5e-4)
        # self.optimizer.setup(self.net.parameters())
        self.totensor = transforms.ToTensor()   
        #self.totensorとself.transformは、データをPyTorchのテンソルに変換するための変換（transform）関数
        #transforms.ToTensor()は、NumPy配列やPIL画像をPyTorchテンソルに変換するための関数. データの前処理に使用
        self.n_action = n_action    #アクションの数を示す変数
        self.count = 0  #トレーニング回数やエポックのカウントを保持する変数
        self.accuracy = 0   #モデルの精度を追跡するための変数
        self.results_train = {} #トレーニングの結果を格納するための辞書
        self.results_train['loss'], self.results_train['accuracy'] = [], [] #トレーニング中の損失と精度の履歴を追跡
        self.loss_list = [] #トレーニング中に計算された損失と精度を記録するためのリスト
        self.acc_list = []  #トレーニング中に計算された損失と精度を記録するためのリスト
        self.datas = [] #トレーニングデータのイメージと対応する目標角度を格納するためのリスト
        self.target_angles = [] #トレーニングデータのイメージと対応する目標角度を格納するためのリスト
        self.criterion = nn.MSELoss()   #損失関数を定義するためのオブジェクト. このコードでは平均二乗誤差（MSELoss）を使用
        self.transform = transforms.Compose([transforms.ToTensor()])
        #self.totensorとself.transformは、データをPyTorchのテンソルに変換するための変換（transform）関数
        #transforms.ToTensor()は、NumPy配列やPIL画像をPyTorchテンソルに変換するための関数. データの前処理に使用
        self.first_flag = True  #フラグ変数で、トレーニングデータの最初のバッチを処理する際に使用
        torch.backends.cudnn.benchmark = False  #GPUでの演算の最適化を制御するための設定. Falseに設定されている場合、最適化は無効になり、再現性が向上
        #self.writer = SummaryWriter(log_dir="/home/haru/nav_ws/src/nav_cloning/runs",comment="log_1")

    def act_and_trains(self, img, target_angle):

        # <training mode>
        self.net.train()

        if self.first_flag:
            self.x_cat = torch.tensor(
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat = self.x_cat.permute(0, 3, 1, 2)
            self.t_cat = torch.tensor(
                [target_angle], dtype=torch.float32, device=self.device).unsqueeze(0)
            self.first_flag = False
        # x= torch.tensor(self.transform(img),dtype=torch.float32, device=self.device).unsqueeze(0)
        # <to tensor img(x),cmd(c),angle(t)>
        x = torch.tensor(img, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        # <(Batch,H,W,Channel) -> (Batch ,Channel, H,W)>
        x = x.permute(0, 3, 1, 2)
        t = torch.tensor([target_angle], dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        self.x_cat = torch.cat([self.x_cat, x], dim=0)
        self.t_cat = torch.cat([self.t_cat, t], dim=0)

        # print(self.x_cat.size()[0])

        # <make dataset>
        #print("train x =",x.shape,x.device,"train c =" ,c.shape,c.device,"tarain t = " ,t.shape,t.device)
        dataset = TensorDataset(self.x_cat, self.t_cat)
        # <dataloder>
        train_dataset = DataLoader(
            dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=True)

        # <only cpu>
        # train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True)

        # <split dataset and to device>
        for x_train, t_train in train_dataset:
            x_train.to(self.device, non_blocking=True)
            t_train.to(self.device, non_blocking=True)
            break

        # <learning>
        # print(t_train)
        self.optimizer.zero_grad()
        # self.net.zero_grad()
        y_train = self.net(x_train)
        # print(y_train,t_train)
        loss = self.criterion(y_train, t_train)
        loss.backward()
        # self.optimizer.zero_grad
        self.optimizer.step()
        # self.writer.add_scalar("loss",loss,self.count)

        # <test>
        self.net.eval()
        action_value_training = self.net(x)
        # self.writer.add_scalar("angle",abs(action_value_training[0][0].item()-target_angle),self.count)
        # print("action=" ,action_value_training[0][0].item() ,"loss=" ,loss.item())
        # print("action=" ,action_value_training.item() ,"loss=" ,loss.item())

        # if self.first_flag:
        #     self.writer.add_graph(self.net,(x,c))
        # self.writer.close()
        # self.writer.flush()
        # <reset dataset>
        if self.x_cat.size()[0] > MAX_DATA:
            self.x_cat = self.x_cat[1:]
            self.t_cat = self.t_cat[1:]
            # self.x_cat = torch.empty(1, 3, 48, 64).to(self.device)
            # self.t_cat = torch.empty(1, 1).to(self.device)
            # self.first_flag = True
            # print("reset dataset")

        # return action_value_training.item(), loss.item()
        return action_value_training[0][0].item(), loss.item()

    def act(self, img):
        self.net.eval()
        # <make img(x_test_ten),cmd(c_test)>
        # x_test_ten = torch.tensor(self.transform(img),dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = torch.tensor(
            img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = x_test_ten.permute(0, 3, 1, 2)
        # print(x_test_ten.shape,x_test_ten.device,c_test.shape,c_test.device)
        # <test phase>
        action_value_test = self.net(x_test_ten)

        print("act = ", action_value_test.item())
        return action_value_test.item()

    def result(self):
        accuracy = self.accuracy
        return accuracy

    def save(self, save_path):
        # <model save>
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        os.makedirs(path)
        torch.save(self.net.state_dict(), path + '/model_gpu.pt')

    def load(self, load_path):
        # <model load>
        self.net.load_state_dict(torch.load(load_path))


if __name__ == '__main__':
    dl = deep_learning()