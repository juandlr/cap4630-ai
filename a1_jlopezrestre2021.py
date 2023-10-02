"""
Project 1 - TicTac Toes using MinMax
Sep 8, 2023
Author: Juan Lopez
Z23635255
"""
import math


def game_init():
    for key in board:
        board[key] = ' '
    print("Positions are as follow:")
    print("1, 2, 3 ")
    print("4, 5, 6 ")
    print("7, 8, 9 ")
    print_board(board)
    while not check_for_win():
        player_move()
        computer_move()


def print_board(game_board):
    print(game_board[1] + '|' + game_board[2] + '|' + game_board[3])
    print('-+-+-')
    print(game_board[4] + '|' + game_board[5] + '|' + game_board[6])
    print('-+-+-')
    print(game_board[7] + '|' + game_board[8] + '|' + game_board[9])
    print("\n")


def space_is_free(position):
    if position in board and board[position] == ' ':
        return True

    return False


def insert_letter(letter, position):
    if space_is_free(position):
        board[position] = letter
        print(f"{letter} turn")
        print_board(board)
        restart = 'n'
        if check_draw():
            print("Draw!")
        if check_for_win():
            if letter == 'X':
                print("You win!")
            else:
                print("AI wins!")

        if check_draw() or check_for_win():
            restart = str(input("Play again y or n:  "))
            if restart == 'y':
                game_init()
            else:
                exit()
        return
    else:
        print("Can't insert there!")
        position = int(input("Please enter new position:  "))
        insert_letter(letter, position)
        return


def check_for_win():
    if board[1] == board[2] and board[1] == board[3] and board[1] != ' ':
        return True
    if board[4] == board[5] and board[4] == board[6] and board[4] != ' ':
        return True
    if board[7] == board[8] and board[7] == board[9] and board[7] != ' ':
        return True
    if board[1] == board[4] and board[1] == board[7] and board[1] != ' ':
        return True
    if board[2] == board[5] and board[2] == board[8] and board[2] != ' ':
        return True
    if board[3] == board[6] and board[3] == board[9] and board[3] != ' ':
        return True
    if board[1] == board[5] and board[1] == board[9] and board[1] != ' ':
        return True
    if board[7] == board[5] and board[7] == board[3] and board[7] != ' ':
        return True

    return False


def check_which_mark_won(mark):
    if board[1] == board[2] and board[1] == board[3] and board[1] == mark:
        return True
    if board[4] == board[5] and board[4] == board[6] and board[4] == mark:
        return True
    if board[7] == board[8] and board[7] == board[9] and board[7] == mark:
        return True
    if board[1] == board[4] and board[1] == board[7] and board[1] == mark:
        return True
    if board[2] == board[5] and board[2] == board[8] and board[2] == mark:
        return True
    if board[3] == board[6] and board[3] == board[9] and board[3] == mark:
        return True
    if board[1] == board[5] and board[1] == board[9] and board[1] == mark:
        return True
    if board[7] == board[5] and board[7] == board[3] and board[7] == mark:
        return True

    return False


def check_draw():
    for key in board.keys():
        if board[key] == ' ':
            return False
    return True


def player_move():
    position = int(input("Enter the position for 'X':  "))
    insert_letter(human, position)
    return


def computer_move():
    """
    Uses the minimax algorithm to make a move that will never lose
    """
    best_score = -math.inf
    best_move = 0

    for key in board.keys():
        if board[key] == ' ':
            board[key] = computer
            score = minimax(board, False, alpha=-math.inf, beta=math.inf)
            board[key] = ' '
            if score > best_score:
                best_score = score
                best_move = key
    print("Number of function calls:", exec_counter)
    insert_letter(computer, best_move)
    return


# Global counter variable
exec_counter = 0


def minimax(board_game, is_maximizing, alpha, beta):
    global exec_counter  # Declare exec_counter as a global variable
    if check_which_mark_won(computer):
        return 1
    elif check_which_mark_won(human):
        return -1
    elif check_draw():
        return 0

    exec_counter += 1  # Increment the counter here

    if is_maximizing:
        best_score = -math.inf
        for key in board_game.keys():
            if board_game[key] == ' ':
                board_game[key] = computer
                score = minimax(board_game, False, alpha, beta)
                board_game[key] = ' '
                if score > best_score:
                    best_score = score
                alpha = max(alpha, best_score)

                # Alpha Beta Pruning
                if beta <= alpha:
                    break
        return best_score
    else:
        best_score = math.inf
        for key in board_game.keys():
            if board_game[key] == ' ':
                board_game[key] = human
                score = minimax(board_game, True, alpha, beta)
                board_game[key] = ' '
                if score < best_score:
                    best_score = score
                beta = min(beta, best_score)
                # Alpha Beta Pruning
                if beta <= alpha:
                    break

        return best_score


board = {1: ' ', 2: ' ', 3: ' ',
         4: ' ', 5: ' ', 6: ' ',
         7: ' ', 8: ' ', 9: ' '}

print("Project 1 - Tic-Tac-Toe")
print("Uses minimax algorithm to simulate a smart tic-tac-toe player")
human = 'X'
computer = 'O'
game_init()
