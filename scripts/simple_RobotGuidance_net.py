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
        self.net.train()    #モデルをトレーニングモードに設定

        if self.first_flag: #最初のデータバッチを処理するかどうかを制御するフラグ．初めてデータバッチが処理される場合（self.first_flagがTrueの場合）、以下の操作が行われる
            self.x_cat = torch.tensor(  #入力画像データimgをテンソルに変換して、self.device（GPUまたはCPU）に配置
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat = self.x_cat.permute(0, 3, 1, 2) #寸法の順番を変更し、チャンネルの次元を先頭に配置
            self.t_cat = torch.tensor(  #ターゲット角度target_angleをテンソルに変換して、self.deviceに配置
                [target_angle], dtype=torch.float32, device=self.device).unsqueeze(0)
            self.first_flag = False #Falseに設定され、以降のデータバッチではこれらの操作はスキップ
        # x= torch.tensor(self.transform(img),dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # <to tensor img(x),cmd(c),angle(t)>    #入力データをテンソルに変換するステップ
        x = torch.tensor(img, dtype=torch.float32,  #imgというNumPy配列またはテンソルをPyTorchテンソルに変換する操作
                         device=self.device).unsqueeze(0)
        
        # <(Batch,H,W,Channel) -> (Batch ,Channel, H,W)>
        x = x.permute(0, 3, 1, 2)   #テンソル x の次元の順序を変更する操作. 通常、PyTorchではチャンネル次元を先頭に配置することが一般的．この操作により、(Batch, H, W, Channel)の形状が(Batch, Channel, H, W)に変更される．これは、モデルが受け入れる形式にデータを整形するためのステップ
        t = torch.tensor([target_angle], dtype=torch.float32,   #目標角度（target_angle）のテンソル表現
                         device=self.device).unsqueeze(0)   #unsqueeze(0)により、バッチ次元を追加
        #self.x_catとself.t_catは、トレーニングデータのバッチを保持するためのテンソル
        #torch.catは、テンソルを結合するための操作で、この場合、既存のデータバッチ（self.x_catとself.t_cat）と新しいデータ（xとt）をバッチ次元（0次元目）で結合. これにより、トレーニングデータバッチが蓄積
        self.x_cat = torch.cat([self.x_cat, x], dim=0)
        self.t_cat = torch.cat([self.t_cat, t], dim=0)

        # print(self.x_cat.size()[0])

        # <make dataset>
        #print("train x =",x.shape,x.device,"train c =" ,c.shape,c.device,"tarain t = " ,t.shape,t.device)
        dataset = TensorDataset(self.x_cat, self.t_cat) #トレーニングデータバッチをPyTorchのデータセットに変換するステップ
        #TensorDatasetは、PyTorchで使用できるデータセットの一種. ミニバッチを作成するのに便利
        
        # <dataloder>
        train_dataset = DataLoader(
            dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'), shuffle=True)
        #DataLoaderは、データセットからバッチを生成し、モデルに供給するための便利なユーティリティ
        #datasetはトレーニングデータセットで、これをDataLoaderに渡す
        #batch_sizeは、1つのミニバッチに含まれるデータポイントの数を指定します。この場合、BATCH_SIZEで指定された値（8）のデータポイントが1つのミニバッチに含まれます。
        #generator=torch.Generator('cpu')は、データローダー内でシャッフルのために使用される乱数生成器を指定しています。'cpu'と指定することでCPU上で乱数生成が行われます。
        #shuffle=Trueは、データをエポックごとにシャッフルするためのフラグです。これにより、トレーニングデータの順序がエポックごとに変更され、モデルのトレーニングが安定化されます。
        
        # <only cpu>
        # train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True)

        # <split dataset and to device>
        for x_train, t_train in train_dataset:  
            #train_datasetからミニバッチごとにデータを取り出すためのループです。一度のループで1つのミニバッチが取り出されます。
            #x_trainは入力データ（画像）のミニバッチを、t_trainは目標データ（目標角度）のミニバッチを表します。
            x_train.to(self.device, non_blocking=True)  #x_trainのデータを計算デバイスに配置するための操作です。
            t_train.to(self.device, non_blocking=True)  #t_trainのデータを計算デバイスに配置するための操作です。
            #non_blocking=Trueは、非同期のデータ移動を有効にするためのオプションです。このオプションを使用することで、データの転送中に他の演算を実行できる場合、効率的なトレーニングが行えます。
            break

        # <learning>
        # print(t_train)
        self.optimizer.zero_grad()
        # self.net.zero_grad()
        #self.optimizer.zero_grad()は、勾配をゼロにリセットするための操作です。モデルのパラメータに関する勾配情報は、前回の勾配計算の結果が残っているため、新しいミニバッチに対して勾配を計算する前にゼロにリセットします。これは、勾配の累積を防ぐための重要なステップです。
        y_train = self.net(x_train)
        # print(y_train,t_train)
        #y_trainは、モデルに入力データ x_train を与えて得られた出力です。モデルは順伝播（forward pass）を実行し、入力から予測結果を計算します。
        loss = self.criterion(y_train, t_train)
        #lossは、予測値 y_train と目標データ t_train の間の損失（誤差）を計算するための操作です。このコードでは、平均二乗誤差（MSELoss）が使用されています。損失関数は、モデルの出力が目標にどれだけ近いかを評価し、その差を数値化します。
        loss.backward()
        #loss.backward()は、逆伝播（backward pass）を実行し、損失関数をモデルのパラメータに関して微分します。これにより、各パラメータに対する勾配が計算されます。
        # self.optimizer.zero_grad
        self.optimizer.step()
        #self.optimizer.step()は、オプティマイザを使用してモデルのパラメータを更新するための操作です。勾配降下法などの最適化アルゴリズムを使用して、モデルの重みを更新し、損失を最小化しようとします。
        # self.writer.add_scalar("loss",loss,self.count)

        # <test>
        self.net.eval() #モデルを評価モードに設定. モデルがデータを順伝播するだけで、重みの更新は行われません。
        action_value_training = self.net(x) #モデルにテスト用の入力データ x を与えて得られた出力です。モデルはテストデータに対して順伝播を実行し、入力から予測結果を計算します。
        # self.writer.add_scalar("angle",abs(action_value_training[0][0].item()-target_angle),self.count)
        # print("action=" ,action_value_training[0][0].item() ,"loss=" ,loss.item())
        # print("action=" ,action_value_training.item() ,"loss=" ,loss.item())

        # if self.first_flag:
        #     self.writer.add_graph(self.net,(x,c))
        # self.writer.close()
        # self.writer.flush()
        # <reset dataset>
        if self.x_cat.size()[0] > MAX_DATA: #現在のトレーニングデータのバッチ数を示します。この値がMAX_DATAを超えた場合、古いデータを削除する処理が実行されます。
            self.x_cat = self.x_cat[1:]
            self.t_cat = self.t_cat[1:]
            #self.x_cat[1:]およびself.t_cat[1:]は、テンソルのスライシングを使用して、最も古いデータポイントを削除しています。この操作により、トレーニングデータのサイズが制御されます。
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
        action_value_test = self.net(x_test_ten)    #action_value_testは、モデルに新しい入力データ x_test_ten を与えて得られた出力です。モデルはテストデータに対して順伝播を実行し、予測結果を計算します。

        print("act = ", action_value_test.item())
        return action_value_test.item() #最終的に、予測された行動値が返されます。

    def result(self):
        accuracy = self.accuracy
        return accuracy

    def save(self, save_path):
        # <model save>
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
        #save_path は、モデルを保存するディレクトリのパスを指定します。このディレクトリにモデルの状態ファイルが保存されます。
        #time.strftime("%Y%m%d_%H:%M:%S") は、現在の日時をフォーマット化した文字列を生成します。この文字列は、モデルの保存ディレクトリの一部として使用されます。
        os.makedirs(path)   #指定されたディレクトリパス (path) を作成します。モデルの保存ディレクトリが存在しない場合に作成します。
        torch.save(self.net.state_dict(), path + '/model_gpu.pt')   #モデルの状態をファイルに保存

    def load(self, load_path):
        # <model load>
        self.net.load_state_dict(torch.load(load_path)) #指定されたパスからモデルの状態を読み込み、self.net にロードする操作です。モデルの状態は torch.load を使用してファイルから復元されます。


if __name__ == '__main__':
    dl = deep_learning()