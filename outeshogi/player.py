class MCTSBot:
    '''
    モンテカルロ木探索で着手を決定するBot
    ストレスの無い消費時間の範囲でで対人対局に臨むには満足な強さにならなかった
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

class Zero:
    '''
    AlphaZeroのアルゴリズムで着手を決定するBot
    学習前はほぼランダム着手に近い
    学習させれば強くなるはず...?
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

class Human:
    '''
    人間がBotと対戦したい時に使う
    '''
    def __init__(self):
        pass
    def move(self, board):
        if board.is_check():
            move = 0  # 相手が勝ち条件を満たしている場合投了する
        else:
            print('指し手を入力してください')
            move = board.move_from_csa(input())
        board.push(move)
        return move

class RandomBot:
    '''
    一様分布からのサンプリング(完全ランダムのこと)で着手を決定するBot
    '''
    def __init__(self, color):
        self.color = color

    def move(self, board):
        if board.is_check():
            move = 0  # 相手が勝ち条件を満たしている場合投了する
        else:
            moves = list(board.legal_moves)
            move = np.random.choice(moves)
        board.push(move)
        return move

class BetterRandomBot:
    '''
    次の一手で勝ちがある時はその手を指し、そうで無い時はランダムに着手を決定するBot
    '''
    def __init__(self):
        pass

    def move(self, board):
        if board.is_check():
            move = 0  # 相手が勝ち条件を満たしている場合投了する
            board.push(move)
        else:
            for move in list(board.legal_moves):
                board.push(move)
                if board.is_check():
                    return move
                else:
                    board.pop()

            moves = list(board.legal_moves)
            move = np.random.choice(moves)
            board.push(move)
            return move
