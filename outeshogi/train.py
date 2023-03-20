SIMULATION_TIMES = 800
EXPLORATION_THRESHOLD = 5
DISPLAY_PUCT = np.inf
DISPLAY_UCT = np.inf
MCTS_THRESHOLD = 10
DISPLAY_COUNT = np.inf
match_nums = 100

board = cshogi.Board()

for match in range(match_nums):
    data = []  # (n, 3)シェイプ。局面の文字列化、その局面での着手、その局の勝敗結果 を記録する。 
    players = [Zero(), Zero()]
    turn = 0

    while True:
        features = board.sfen()
        move = players[turn].move(board)
        move = make_move_label(move, board.turn)
        result = None
        data.append([features, move, result])
        #print(board)  # コメント解除すると1手進むたびに盤面を表示する
        if (board.is_check()):
            if board.turn == 0:
                print(f'{match+1}局目：後手の勝ちです')
                result = 0  # 白の勝ち
            else:
                print(f'{match+1}局目：先手の勝ちです')
                result = 1  # 黒の勝ち
            break
        elif (board.is_draw()):
            print(f'{match+1}局目：千日手で引き分けです')
            result = 0.5
            break
        turn = 1 - turn
    board.reset()

    # 教師データにするので対局結果を保存する
    data = np.array(data)
    data[0::2, 2] = result  # 先手の手番の局面には結果をそのまま入れる
    data[1::2, 2] = 1 - result # 後手の手番の局面は、結果を逆にして入れる
    if match > 0:  # 一局目の結果はvstackせずそのまま
        all_data = np.vstack((all_data, data))
    else:
        all_data = data
    
    with open('パス.pickle', 'wb') as f:  # ノートブックで実行する時にGoogleDriveをマウントしておけば結果を好きなパスに保存できる
        pickle.dump(all_data, f)
