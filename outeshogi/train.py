from torch.utils.data import Dataset

from time import time

class MyDataset(Dataset):
    '''
    play.pyで作った対局結果をDataSet化する
    '''
    def __init__(self, data):
        self.features = data[:, 0]
        self.labels1 = data[:, 1]
        self.labels2 = data[:, 2]

    def __len__(self):
        return len(self.features)
  
    def __getitem__(self, idx):
        '''
        feature : 盤面の特徴量
        label1 : 全ての着手分の配列(2172, )のうち、実際の着手を1, それ以外を全て0にした配列
        label2 : その盤面における手番側視点の勝率
        '''
        board = cshogi.Board(self.features[idx])
        feature = np.zeros((FEATURES_NUM, 9, 9), dtype=np.float32)
        make_input_features(board, feature)
        feature = torch.tensor(feature, dtype=torch.float32)

        label1 = np.zeros(MOVE_LABELS_NUM)
        move_label = self.labels1[idx]
        label1[move_label] = 1.0
        label1 = np.array(label1, dtype=np.float32)

        label2 = np.array(self.labels2[idx], dtype=np.float32)[np.newaxis]

        return feature, label1, label2

# GPUの確認
use_cuda = torch.cuda.is_available()
print('Use CUDA:', use_cuda)

model = PolicyValueNetwork()
#model = torch.load('モデルパス')  # 保存してあるモデルを使用する場合はここにパスを入れる 

if use_cuda:
    model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=0.0001)

# ミニバッチサイズ・エポック数の設定
batch_size = 4096
epoch_num = 5
with open('教師データパス', 'rb') as f:  # ここに教師データのパスを入れる
    data = pickle.load(f)
train_data = MyDataset(data)
n_iter = len(train_data) // batch_size

# データローダーの設定
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

# 誤差関数の設定
policy_criterion = nn.CrossEntropyLoss()
value_criterion = nn.BCEWithLogitsLoss()
lambda_1 = 1.
lambda_2 = 1.
if use_cuda:
    policy_criterion.cuda()
    value_criterion.cuda()

start = time()
for epoch in range(1, epoch_num+1):
    sum_value_loss = 0.0
    sum_policy_loss = 0.0
    sum_all_loss = 0.0
    cls_count = 0
    reg_count = 0
    num_data = 0
    
    for feature, label1, label2 in train_loader:
        if use_cuda:
            feature = feature.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()

        policy, value = model(feature)

        policy_loss = policy_criterion(policy, label1)
        value_loss = value_criterion(value, label2)
        all_loss = lambda_1 * policy_loss + lambda_2 * value_loss 

        model.zero_grad()
        all_loss.backward()
        optimizer.step()
        
        sum_policy_loss += policy_loss.item()
        sum_value_loss += value_loss.item()
        sum_all_loss += all_loss.item()
        
    print("epoch: {}, mean loss: {}, mean loss(value): {}, mean loss(policy): {}, elapsed_time :{}".format(epoch,
                                                                                 sum_all_loss / n_iter,
                                                                                 sum_value_loss / n_iter,
                                                                                 sum_policy_loss / n_iter,
                                                                                 time() - start))
print("Done.")
