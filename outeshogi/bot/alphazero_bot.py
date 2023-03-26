C_BASE = 19652
C_INIT = 1.25
DIRICHLET_ALPHA = 0.35
DIRICHLET_EPS = 0.25

class ZeroParameters:
    '''モンテカルロ木探索用のパラメーターN(訪問回数)とW(累積報酬)とP(着手確率)'''
    def __init__(self):
        self.N = {}
        self.W = {}
        self.P = {}

class AlphaZeroBot:
    '''
    AlphaZeroアルゴリズムで着手を決定するBot
    '''
    def __init__(self):
        self.params = ZeroParameters()
        self.alphazero = AlphaZero(self.params)

    def move(self, board):
        if board.is_check():
            move = 0  # 相手が勝ち条件を満たしている場合投了する
        else:
            move = self.alphazero.search(board)
        board.push(move)
        return move
      
class AlphaZero:
    def __init__(self, params):
        self.params = params
        self.model = PolicyValueNetwork()
        #self.model = torch.load('モデルのパス')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def create_features(self, board):
        '''
        引数で渡ってきたboardの状況を、ニューラルネットワークへ入力できる形式に変換する
        '''
        features = np.zeros((FEATURES_NUM, 9, 9))
        make_input_features(board, features)
        features = features[np.newaxis, :, :, :]
        x = torch.tensor(features, dtype=torch.float32)
        return x

    def infer(self, x):
        '''
        ニューラルネットワークによる推測結果を返す
        -----------------------------------
        返り値
        policy.shape : [2172, ]
        value : (0, 1)の勝率を表す数値
        '''
        x = x.to(self.device)
        with torch.no_grad():
            policy, value = self.model(x)
        policy = policy.cpu()
        value = value.cpu()
        policy = policy[0].numpy() # policyは(1, 2172)shapeの二次元配列
        value = torch.sigmoid(value).item()
        return policy, value  
    
    def make_legalmove_probs(self, board, policy):
        '''
        policy(.shape==[2172, ]の配列)を渡すと
        boardにおける有効手以外を0にし、かつそれ以外の合計を1にして返す
        '''
        legal_moves = list(board.legal_moves)
        legal_move_probs = np.zeros_like(policy, dtype=np.float32)  # 全て0の配列を作って埋めていく
        for move in legal_moves:
            move_label = make_move_label(move, board.turn)
            legal_move_probs[move_label] = policy[move_label]  # legal_moveのlabelと同じ位置に限って、policyの値で埋めていく
        max_prob = max(legal_move_probs[legal_move_probs != 0])
        min_prob = min(legal_move_probs[legal_move_probs != 0])
        legal_move_probs[legal_move_probs!=0] -= min_prob
        legal_move_probs /= (max_prob - min_prob)
        legal_move_probs /= np.sum(legal_move_probs)  # 確率の合計が1になるようにする
        return legal_move_probs

    def search(self, board):
        '''
        渡ってきたboardにおけるNを参照して行動を決定する
        '''
        self.puct_count = 0  # デバッグ用
        
        root_hash = board.zobrist_hash()
        if root_hash not in self.params.P:  # 子盤面が無い場合は展開する
            self._expand(board)

        # ルート局面のPにディリクレノイズをのせて探索を促進する
        legal_moves = list(board.legal_moves)
        dirichlet_noise = np.random.dirichlet(alpha=[DIRICHLET_ALPHA]*len(legal_moves))
        for move, noise in zip(legal_moves, dirichlet_noise):
            move_label = make_move_label(move, board.turn)
            self.params.P[root_hash][move_label] = (1 - DIRICHLET_EPS) * self.params.P[root_hash][move_label] + DIRICHLET_EPS * noise
        
        for simulation in range(SIMULATION_TIMES):  SIMULATION_TIMES回シミュレーションする
            self._simulate(board)

            if (simulation+1) % DISPLAY_COUNT == 0:
                # デバッグ用にparams.Nと指し手の組み合わせの途中経過を表示
                legal_move_labels = make_legalmove_labels(board)
                lst = []
                for label in legal_move_labels:
                    lst.append([self.params.N[root_hash][label], cshogi.move_to_csa(move_from_label(board, label))])
                print(simulation+1)
                print(sorted(lst, reverse=True))

        if board.move_number < EXPLORATION_THRESHOLD:  # 閾値までは探索回数を確率分布として扱う。閾値を超えたらNの最大値を着手として選択する。
            policy = np.array(self.params.N[root_hash]) / np.sum(self.params.N[root_hash])
            move_label = np.random.choice(range(MOVE_LABELS_NUM), p=policy)
            move = move_from_label(board, move_label)
            print(f'{board.move_number}手目:Nを確率分布として{cshogi.move_to_csa(move)}を選択。')  # デバッグ
        else:
            ns = np.array(self.params.N[root_hash])
            move_label = np.random.choice(np.where(ns==max(ns))[0])
            move = move_from_label(board, move_label)
            print(f'{board.move_number}手目:Nの最大値として{cshogi.move_to_csa(move)}を選択。')  # デバッグ
        # デバッグ用にparams.Nと指し手の組み合わせを表示
        legal_move_labels = make_legalmove_labels(board)
        lst = []
        for label in legal_move_labels:
            lst.append([self.params.N[root_hash][label], cshogi.move_to_csa(move_from_label(board, label))])
        print(sorted(lst, reverse=True))
        return move  
    
    def _expand(self, board):
        '''
        ①渡ってきた局面をルートとし、子ノードへのリンクを作成する
        ②渡ってきた局面をニューラルネットワークで評価して評価値を返す
        '''
        # ニューラルネットでルート局面を評価
        x = self.create_features(board)
        policy, value = self.infer(x)
        legal_move_policy = self.make_legalmove_probs(board, policy)

        hash = board.zobrist_hash()
        self.params.P[hash] = legal_move_policy
        self.params.N[hash] = [0] * MOVE_LABELS_NUM
        self.params.W[hash] = [0] * MOVE_LABELS_NUM
        
        return value
    
    def _simulate(self, board):
        '''
        ルート局面からPUCT値最大の行動を実行して評価を更新する
        '''
        hash = board.zobrist_hash()

        # PUCT値を計算して最大値の行動を実行
        puct_list = np.array(self._calc_puct(board))
        move_label = np.random.choice(np.where(puct_list==max(puct_list))[0])
        move = move_from_label(board, move_label)
        next_board = board.copy()
        next_board.push(move)

        # 動かした盤面を評価する。相手の手番なので報酬は逆にする
        v = 1 - self._evaluate(next_board)

        self.params.W[hash][move_label] += v
        self.params.N[hash][move_label] += 1

    def _calc_puct(self, board):
        self.puct_count += 1  # デバッグ用
        '''
        渡された場面からの全ての着手のPUCT値リストを返す(非合法手は-np.inf)
        '''
        hash = board.zobrist_hash()
        N = np.sum(self.params.N[hash])
        puct_list = []
        legal_move_labels = make_legalmove_labels(board)

        for label in range(MOVE_LABELS_NUM):  # 全ての子ノードのPUCT値を求める(非合法手は-np.inf)
            if label in legal_move_labels:
                # PUCT値を計算
                n = self.params.N[hash][label]
                w = self.params.W[hash][label]
                p = self.params.P[hash][label]
                c = np.log((1 + N + C_BASE) / C_BASE) + C_INIT
                U = c * p * np.sqrt(N) / (1 + n)
                Q = w / n if n != 0 else 0
                puct = U + Q
            else:
                puct = -np.inf
            puct_list.append(puct)
        
        # デバッグ用にpuctと指し手の組み合わせを表示
        if self.puct_count % DISPLAY_PUCT == 0:
            print(self.puct_count)
            legal_moves = list(board.legal_moves)
            puct_dict = {}
            for move in legal_moves:
                move_label = make_move_label(move, board.turn)
                puct = puct_list[move_label]
                puct_dict[cshogi.move_to_csa(move)] = puct
            print(sorted(puct_dict.items(), key=lambda x:x[1], reverse=True))
        return puct_list
    
    def _evaluate(self, board):
        '''
        渡された盤面を評価する
        '''
        hash = board.zobrist_hash()
        if board.is_check() or board.is_draw():  # 終局している場合はその時点で報酬を算出
            reward = self._make_reward(board, board.turn)
            return reward
        elif not hash in self.params.P:  # ノードが未展開の場合は展開してこの盤面だけ評価する
            value = self._expand(board)
            #value = self._rollout(board, board.turn)
            return value
        else:
            # 終局しておらずノードが展開済みの場合はPUCTによってさらに子盤面に読み進める
            puct_list = np.array(self._calc_puct(board))
            move_label = np.random.choice(np.where(puct_list==max(puct_list))[0])
            move = move_from_label(board, move_label)
            next_board = board.copy()
            next_board.push(move)

            # 進めた局面を評価する(相手の手番になるので報酬は逆にする)
            v = 1 - self._evaluate(next_board)

            self.params.W[hash][move_label] += v
            self.params.N[hash][move_label] += 1

            return v
            
    def _make_reward(self, board, turn):
        '''
        turn視点の報酬を返す
        黒:0 白:1
        '''
        if board.is_check():
            if board.turn == BLACK:  # 黒が負けの場合
                return turn
            else:  # 白が負けの場合
                return 1 - turn
        if board.is_draw():  # 引き分けの場合
            return 0.5
