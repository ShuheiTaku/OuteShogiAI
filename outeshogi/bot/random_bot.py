class RandomBot:
    '''
    完全ランダムで着手を決定するBot。MCTSのロールプレイなどで使える。
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

class BetterRandomBot:
    '''
    次の1手で勝ちになる手がある盤面では必ずその手を指し、それ以外はランダムで着手するBot
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
