class MCTSParameters:
    '''モンテカルロ木探索用のパラメーターN(訪問回数)とW(累積報酬)'''
    def __init__(self):
        self.N = {}
        self.W = {}
        
class MCTSBot:
    '''
    モンテカルロ木探索で着手を決定するBot
    '''
    def __init__(self, params):
        self.params = params
        self.mcts = MCTS(self.params)

    def move(self, board):
        if board.is_check():
            move = 0  # 相手が勝ち条件を満たしている場合投了する
        else:
            move = self.mcts.search(board)
        board.push(move)
        return move

class MCTS:
    def __init__(self, params):
        self.params = params

    def search(self, board):
        '''
        渡ってきたboardにおけるNを参照して行動を決定する
        '''
        self.uct_count = 0

        root_hash = board.zobrist_hash()
        if root_hash not in self.params.N:  # 子盤面が無い場合は展開する
            self._expand(board)
        
        for simulation in range(SIMULATION_TIMES_MCTS):
            self._simulate(board)
            if (simulation + 1) % DISPLAY_COUNT == 0:
                # デバッグ用にparams.Nと指し手の組み合わせを表示
                legal_move_labels = make_legalmove_labels(board)
                lst = []
                for label in legal_move_labels:
                    lst.append([self.params.N[root_hash][label], cshogi.move_to_csa(move_from_label(board, label))])
                print(f'simulation:{simulation+1}')
                print(sorted(lst, reverse=True))


        if board.move_number < EXPLORATION_THRESHOLD:  # 閾値までは探索回数を確率分布として扱う
            policy = np.array(self.params.N[root_hash]) / np.sum(self.params.N[root_hash])
            move_label = np.random.choice(range(MOVE_LABELS_NUM), p=policy)
            move = move_from_label(board, move_label)
            print(f'Nを確率分布として{cshogi.move_to_csa(move)}を選択。')  # デバッグ
        else:
            ns = np.array(self.params.N[root_hash])
            move_label = np.random.choice(np.where(ns==max(ns))[0])
            move = move_from_label(board, move_label)
            print(f'Nの最大値として{cshogi.move_to_csa(move)}を選択。')  # デバッグ
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
        hash = board.zobrist_hash()
        self.params.N[hash] = [0] * MOVE_LABELS_NUM
        self.params.W[hash] = [0] * MOVE_LABELS_NUM

    def _simulate(self, board):
        '''
        ルート局面からUCT値最大の行動を実行して評価を更新する
        '''
        hash = board.zobrist_hash()

        # UCT値を計算して最大値の行動を実行
        uct_list = np.array(self._calc_uct(board))
        move_label = np.random.choice(np.where(uct_list==max(uct_list))[0])
        move = move_from_label(board, move_label)
        next_board = board.copy()
        next_board.push(move)

        # 動かした盤面を評価する。相手の手番なので報酬は逆にする
        count = self.params.N[hash][move_label]
        v = -self._evaluate(next_board , count)

        self.params.W[hash][move_label] += v
        self.params.N[hash][move_label] += 1

    def _calc_uct(self, board):
        self.uct_count += 1
        '''
        渡された場面からの全ての着手のUCT値リストを返す(非合法手は-np.inf)
        '''
        hash = board.zobrist_hash()
        N = np.sum(self.params.N[hash])
        uct_list = []
        legal_move_labels = make_legalmove_labels(board)

        for label in range(MOVE_LABELS_NUM):  # 全ての子ノードのUCT値を求める(非合法手は-np.inf)
            if label in legal_move_labels:
                # UCT値を計算
                n = self.params.N[hash][label]
                w = self.params.W[hash][label]
                c = np.sqrt(2)
                if n == 0:
                    uct = np.inf  # 全ての行動を最低1度は選択する
                else:
                    U = c * np.sqrt(np.log(N) / (n))
                    Q = w / n
                    uct = U + Q
            else:
                uct = -np.inf
            uct_list.append(uct)
        
        # デバッグ用にuctと指し手の組み合わせを表示
        if self.uct_count % DISPLAY_UCT == 0:
            print(self.uct_count)
            legal_moves = list(board.legal_moves)
            uct_dict = {}
            for move in legal_moves:
                move_label = make_move_label(move, board.turn)
                uct = uct_list[move_label]
                uct_dict[cshogi.move_to_csa(move)] = uct
            print(sorted(uct_dict.items(), key=lambda x:x[1], reverse=True))
        return uct_list
    
    def _evaluate(self, board, visit_count):
        '''
        渡された盤面を評価する
        '''
        hash = board.zobrist_hash()
        if board.is_check() or board.is_draw():  # 終局している場合はその時点で報酬を算出
            reward = self._make_reward(board, board.turn)
            return reward
        elif hash not in self.params.N :  # 未展開の場合はまず展開してこの局面の評価を返す
            self._expand(board)
            value = self._rollout(board, board.turn)
            return value
        elif visit_count < MCTS_THRESHOLD:  # 訪問回数が少ないのでロールアウト
            value = self._rollout(board, board.turn)
            return value
        else:  # この局面の訪問回数が閾値を超えている場合は先読みする
            uct_list = np.array(self._calc_uct(board))
            move_label = np.random.choice(np.where(uct_list==max(uct_list))[0])
            move = move_from_label(board, move_label)
            next_board = board.copy()
            next_board.push(move)

            # 進めた局面を評価する(相手の手番になるので報酬は逆にする)
            count = self.params.N[hash][move_label]
            v = -self._evaluate(next_board, count)
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
                reward = -1 if turn == 0 else 1
                return reward
            else:  # 白が負けの場合
                reward = 1 if turn == 0 else -1
                return reward
        if board.is_draw():  # 引き分けの場合
            return 0
    
    # ロールアウト
    def _rollout(self, board, turn):
        players = [RandomBot(0), RandomBot(1)]
        end = False
        while not end:
            turn = board.turn
            players[turn].move(board)
            if (board.is_check()) or (board.is_draw()):
                end = True

        reward = self._make_reward(board, turn)
        return reward
