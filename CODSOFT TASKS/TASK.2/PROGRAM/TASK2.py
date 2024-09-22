import math

def print_board(board):
    for row in board:
        print("|".join(row))
        print("-" * 5)

def check_winner(board, player):
    for row in board:
        if row.count(player) == 3:
            return True
    for col in range(3):
        if [board[row][col] for row in range(3)].count(player) == 3:
            return True
    if [board[i][i] for i in range(3)].count(player) == 3 or [board[i][2 - i] for i in range(3)].count(player) == 3:
        return True
    return False

def is_board_full(board):
    return all(cell != " " for row in board for cell in row)

def minimax(board, depth, is_maximizing):
    if check_winner(board, "O"):
        return 1  
    elif check_winner(board, "X"):
        return -1  
    elif is_board_full(board):
        return 0  

    if is_maximizing:
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "O"
                    score = minimax(board, depth + 1, False)
                    board[i][j] = " "
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "X"
                    score = minimax(board, depth + 1, True)
                    board[i][j] = " "
                    best_score = min(score, best_score)
        return best_score

def find_best_move(board):
    best_move = None
    best_score = -math.inf
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board[i][j] = "O"
                score = minimax(board, 0, False)
                board[i][j] = " "
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
    return best_move

def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]
   
    print("Welcome to Tic-Tac-Toe! You are X, and the AI is O.")
   
    while True:
        print_board(board)
       
        while True:
            row = int(input("Enter your move row (0, 1, or 2): "))
            col = int(input("Enter your move column (0, 1, or 2): "))
            if board[row][col] == " ":
                board[row][col] = "X"
                break
            else:
                print("Invalid move. Try again.")
       
        if check_winner(board, "X"):
            print_board(board)
            print("Congratulations! You win!")
            break
        if is_board_full(board):
            print_board(board)
            print("It's a draw!")
            break
       
        print("AI is making a move...")
        move = find_best_move(board)
        if move:
            board[move[0]][move[1]] = "O"
       
        if check_winner(board, "O"):
            print_board(board)
            print("AI wins! Better luck next time.")
            break
        if is_board_full(board):
            print_board(board)
            print("It's a draw!")
            break

play_game()