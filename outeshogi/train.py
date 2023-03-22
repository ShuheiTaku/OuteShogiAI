from torch.utils.data import Dataset

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
