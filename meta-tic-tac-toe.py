# -*- coding: utf-8 -*-
"""
Meta noughts and crosses

Created on Tue Aug 28 17:48:38 2018

@author: rn7318

board = the whole playing field.
grid = a section of the board containing 1 noughts-and-crosses grid
cell = a square in a grid; space for 1 number.
"""

import numpy as np, matplotlib.pyplot as plt

#def check_winner(board):
#    winner = 0  # assume no winner
#    rows = [board[ii, :] for ii in range(3)]
#    cols = [board[:, ii] for ii in range(3)]
#    diags = [np.diag(board), np.array([board[-1-ii, ii] for ii in range(3)])]
#    lines_to_check = rows + cols + diags
#    for l in lines_to_check:
#        # if all line elements are not equal, skip this line.
#        if not np.all(l == l[0]):
#            continue
#        elif l[0] == 0:  # if all elements equal zero, skip this line
#            continue
#        else:
#            winner = l[0]
#    return winner

""" https://stackoverflow.com/questions/39181600/find-all-n-dimensional-lines-and-diagonals-with-numpy """

class TicTacToe():
    def __init__(self, board=None, size=3):

        if board is None:   # if only "size" passed; construct board
            self.size = size
            self.board = np.zeros((size, size)).astype(int)
        else:  # if only board passed; calculate size
            self.board = board
            self.size = np.sqrt(len(board.flat)).astype(int)

        self.current_player = self.get_current_player()

        # store references to board slices to speed up checking later
        self.rows = [np.s_[ii, :] for ii in range(size)]
        self.cols = [np.s_[:, ii] for ii in range(size)]
        r = list(range(size))
        self.diags = [np.s_[r, r], np.s_[r[::-1], r]]
        self.slices = self.rows + self.cols + self.diags

    def get_current_player(self):
        board = self.board
        if np.all(board == 0):  # if the board is empty
            return 1  # noughts to start
        count_o = list(board.flat).count(1)
        count_x = list(board.flat).count(2)
        if count_o <= count_x:
            return 1  # noughts to play next
        else:
            return 2

    def xo(self, c):  # function to convert 0,1 to O,X string
        if c==0:
            return "."
        elif c==1:
            return "o"
        elif c==2:
            return "x"
        else:
            return "?"

    def print_board(self):
        board = self.board
        # convert 1/2 to x/o and convert 0 to position number
        board = [(self.xo(b) if b != 0 else i) for i, b in enumerate(board.flat)]
        board = np.reshape(board, (self.size, -1))  # vector board to matrix
        for row in board:
            line = " ".join(["{:<2}",]*self.size)
            print(line.format(*row))

    def play(self, index, value):
        if index not in self.legal_moves():
            raise Exception("you can't play in slot {}".format(index))
        if value != self.get_current_player():
            raise Exception("it's not player {}'s turn!".format(value))
        self.board.flat[index] = value  # update the value on the board

    def check_winner(self):

        winner = 0  # assume no winner
        if np.all(self.board == 0):  # if the board is empty
            return winner

        for s in self.slices:  # check each slice; rows, cols, diags
            line = self.board[s]  # get values on current slice
            # if all line elements are not equal, skip this line.
            if not np.all(line == line[0]):
                continue
            elif line[0] == 0:  # if all elements equal zero, skip this line
                continue
            else:
                winner = line[0]
                break
        return winner

    def legal_moves(self):
        return [i for i, b in enumerate(self.board.flat) if b == 0]

    def copy(self):  # so that the algorithm can make hypothetical moves
        return TicTacToe(self.board.copy())

class MetaTicTacToe(TicTacToe):
    def __init__(self, board=None, size=3, last_move=None):

        # create the board of sub-grids
        if board is None:   # if only "size" passed; construct board
            self.size = size
            self.board = np.zeros((size**2, size**2)).astype(int)
        else:  # if only board passed; calculate size
            self.board = board
            self.size = np.sqrt(np.sqrt(len(board.flat))).astype(int)

        # create the "meta-board" representing the winner of each sub-grid
        self.meta_board = np.zeros((size, size)).astype(int)

        self.current_player = self.get_current_player()  # store current player

        # store references to board slices to speed up checking later
        self.rows = [np.s_[ii, :] for ii in range(size)]
        self.cols = [np.s_[:, ii] for ii in range(size)]
        r = list(range(size))
        self.diags = [np.s_[r, r], np.s_[r[::-1], r]]
        self.slices = self.rows + self.cols + self.diags

        self.last_move = last_move  # store last played move internally

        # store references to sub-grids
        self.grid_masks = np.array([[None for i in range(size)] for j in range(size)])
        for I in range(size):
            for J in range(size):
                mask = np.s_[I*size:I*size+size, J*size:J*size+size]
                self.grid_masks[I][J] = mask

    def print_board(self):
        board = []
        size = self.size
        # convert 1/2 to x/o and convert 0 to position number
        for i, b in enumerate(self.board.flat):
            if b != 0:  # for occupied squares
                board.append(self.xo(b))  # place xo string
            else:  # if space is empty
                if i in self.legal_moves():
                    board.append(i)  # number only the legal moves
                else:
                    board.append(".")  # put dots for the rest
        board = np.reshape(board, (self.size**2, -1))  # vector board to matrix
        for ii, row in enumerate(board):
            quad = " ".join(["{:<3}",]*size)  # template for numbers in a quad
            line = " | ".join([quad,]*size)  # template for one line of board
            if (ii % size == 0) and (ii != 0):
                print(line.format(*["--"]*size**2))
            print(line.format(*row))

    def check_grid_winner(self, grid):

        winner = 0  # assume no winner
        if np.all(grid == 0):  # if the board is empty
            return winner

        for s in self.slices:  # check each slice; rows, cols, diags
            line = grid[s]  # get values on current slice
            # if all line elements are not equal, skip this line.
            if not np.all(line == line[0]):
                continue
            elif line[0] == 0:  # if all elements equal zero, skip this line
                continue
            else:
                winner = line[0]
                break

        # if no winner but the grid is full, set winner to None
        if (winner == 0) and np.all(grid != 0):
            winner = -1  # -1 means draw

        return winner

    def check_winner(self):
        # check the winner of every sub-grid and update the meta-board
        for ii, mask in enumerate(self.grid_masks.flat):
            grid = self.board[mask]
            self.meta_board.flat[ii] = self.check_grid_winner(grid)
        # check the meta-board for a winner
        return self.check_grid_winner(self.meta_board)

    def play(self, index, value):
        super().play(index, value)
        self.last_move = index

    def legal_moves(self):

        if self.last_move is not None:
            k = self.last_move
            a, b = self.k_to_ab(k)
            conquered_grid = self.meta_board[a, b] != 0
        else:
            conquered_grid = False

        # at the start of the game or if previous move sent player to conquered grid
        if (self.last_move is None) or conquered_grid:
            return [k for k, b in enumerate(self.board.flat) if b == 0]
        else:
            # get the last move and identify what position it was in in its sub-grid.
            I, J = self.k_to_ab(self.last_move)
            k_list = self.IJ_to_k(I, J)  # get the list of numbers in that grid
            return [k for k in k_list if self.board.flat[k] == 0]

    def copy(self):  # so that the algorithm can make hypothetical moves
        return MetaTicTacToe(board=self.board.copy(),
                             grid_masks = np.array([[None for i in range(size)] for j in range(size)]),
                             last_move=self.last_move)

    def ij_to_IJ(self, i, j):
        """ i, j are the coordinates of an individual square (in matrix coords)
        i.e. i = row, j = col. I, J are the coordinates of each sub-grid on the
        meta-board """
        size = self.size
        return i // size, j // size

    def IJ_to_ij(self, I, J):
        size = self.size
        i_range = range(I*size, (I+1)*size)
        j_range = range(J*size, (J+1)*size)
        return [(i, j) for i in i_range for j in j_range]

    def IJ_to_k(self, I, J):
        k_list = [self.ij_to_k(i, j) for i, j in self.IJ_to_ij(I, J)]
        return k_list

    def k_to_ij(self, k):
        """ k is the index of each square in the flattened board (this is the
        number the user enters to play a move). i, j are the matrix coords of
        the square """
        size = self.size
        i, j = divmod(k, size**2)
        return i, j

    def ij_to_k(self, i, j):
        size = self.size
        return i * size**2 + j % size**2

    def k_to_IJ(self, k):
        return self.ij_to_IJ(*self.k_to_ij)

    def k_to_ab(self, k):
        """ k is flat index of each square. a, b are the matrix coords of the
        square relative to its sub-grid. This is used to identify which sub-
        grid becomes active after a move is made """
        size = self.size
        a = k // size**2 % size
        b = k % size
        return a, b

#game = MetaTicTacToe(size=3)
#game.play(40, 1)
#game.play(30, 2)
#game.play(1, 1)
#game.play(3, 2)
#game.play(19, 1)
#game.play(57, 2)
#game.play(10, 1)
#game.print_board()
#
#game.board.flat[31] = 2
#game.board.flat[32] = 1
#game.board.flat[39] = 1
#game.board.flat[41] = 2
#game.board.flat[48] = 2
#game.board.flat[49] = 2
#game.board.flat[50] = 1
#game.print_board()
#game.check_global_winner()
#game.meta_board


#%%

'''
class Brute():
    def __init__(self, game, think_N_rounds_ahead=2, identity=1):
        print("ALGORITHM INITIALISING. PREPARE TO HAVE YOUR ASS WHIPPED")
        self.game = game  # store the instance of the game it's playing
        self.think_N_rounds_ahead = think_N_rounds_ahead
        self.identity = identity  # the side to play as (1 = O, 2 = X)
        self.opponent = 2 if identity == 1 else 1  # store opposite as opponent

    def plan_next_move(self, game=None, playing_as=None, level=0):
        """ This function recursively checks N moves ahead, and returns the
        best option based on the number of wins/losses in the "tree" of options
        ahead """

        if playing_as is None:  # if not told who to play as
            playing_as = self.identity  # play as self
        # opponent is the opposite of whoever we're playing as right now
        opponent = self.opponent if playing_as == self.identity else self.identity

        if game is None:
            game = self.game  # operate on the main game unless otherwise specified

        winner = game.check_winner()  # check if this move ended the game
        if winner != 0:  # if there is a winner, stop and return score
            return dict(winner=winner)  # return the winner
            # I have to use a dict to differentiate between winner and score

        # if there's no winner, but we have reached the recursion limit
        if level == self.think_N_rounds_ahead*2:  # 2 moves in a round
            return dict(score=0)  # return zero for no/unknown winner

        # if no winner and not yet at the recursion limit
        # make a list of the available moves -- i.e. the empty squares
        moves = game.legal_moves()
        if len(moves) == 0:  # if there are no more moves left to make
            return dict(score=0)  # return the score for a stalemate

        # evaluate each of these moves, and get the sum of their scores
        outcomes = []
        for move in moves:
            # see what the board will look like if you make that move
            new_game = game.copy()
            new_game.play(move, playing_as)
            # then plan the opponent's next moves
            o = self.plan_next_move(new_game, playing_as=opponent,
                                          level=level+1)
            outcomes.append(o)

            # if the move causes self to win the game
            if o.get("winner") == playing_as:
                # stop the analysis; current player only needs one win option to win.
                break

        winners = [o.get("winner") for o in outcomes]
        scores = [o.get("score") for o in outcomes]

        if level == 0:
            print("I HAVE EVALUATED THE FOLLOWING MOVES:")
            template = "{:<10} {:<10} {:<10} {:<10}"
            print(template.format("index", "moves", "winners", "scores"))
            for i, (m, w, s) in enumerate(zip(moves, winners, scores)):
                s = "None" if s is None else s
                w = "None" if w is None else w
                print(template.format(i, m, w, s))

        # decide what to output
        # if ANY of the moves cause current player to win,
        if playing_as in winners:
            output = dict(winner=playing_as)  # pass on result to parent move
        # else if ALL of the moves cause a win for current opponent
        elif np.all([w == opponent for w in winners]):
            output = dict(winner=opponent)  # pass on the result to parent move
        else:  # if the parent move is inconclusive
            # calculate the score instead of returning a winner:
            # -1 for every win for the opponent, plus the scores for the undecided games
            s = (sum(o["score"] for o in outcomes if "score" in o)
                      -1 * winners.count(opponent))
            output = dict(score=s)

        if level > 0:  # if this is some intermediate level
            return output  # just return the winner or scores for this node

        # if this is the top level caller, return the best option for the algorithm
        elif level == 0:
            # if the algorithm has an option to win
            if playing_as in winners:
                index = winners.index(playing_as)
                move = moves[index]
                return dict(move=move, confidence=True)
            # if the opponent will win no matter what move the algo makes
            elif np.all([w == opponent for w in winners]):
                return dict(move=moves[0], confidence=False)
            else:  # else just pick the best move
                max_score = max(o["score"] for o in outcomes if "score" in o)
                index = scores.index(max_score)
                move = moves[index]
                return dict(move=move, confidence=max_score)

def print_winner(winner):
    if winner is 0:
        print("it's a draw")
    else:
        print("the winner is ", game.xo(winner))
game = MetaTicTacToe(size=2)
algo = Brute(game, think_N_rounds_ahead=1, identity=2)
legal_moves = game.legal_moves()
human_move = None
game.print_board()

while len(legal_moves) > 0:

    # human move
    while human_move not in legal_moves:
        inp = input("choose your move, human\n")
        if inp == "stop":
            raise Exception()
        human_move = int(inp)
    game.play(human_move, 1)
    game.print_board()
    legal_moves = game.legal_moves()
    print("the allowed moves are now:",legal_moves)

    winner = game.check_winner()
    if winner or not legal_moves:
        print_winner(winner)
        break

    # algorithm move
    out = algo.plan_next_move()
    algo_move, confidence = out["move"], out["confidence"]
    print("algo moves to ",algo_move,"and is",confidence,"confident about it")
    if algo_move not in legal_moves:
        raise Exception("algorithm made an illegal move!")
    game.play(algo_move, algo.identity)
    game.print_board()
    legal_moves = game.legal_moves()
    print("the allowed moves are now:",legal_moves)

    winner = game.check_winner()
    if winner or not legal_moves:
        print_winner(winner)
        break

#'''

""" fix legal_moves so that it can tell if the player has been sent to a
conquered grid! or a full grid! """
# %%
"""
NOTES:
    make it less defeatist when it detects that it has definitely lost. Don't
    have it choose just a random option. Have it minimise the damage somehow.
    rank the losses by how soon they occur?

new algorithm:
    - make a list of all the possible next moves
    - evaluate each of these moves.
        - if a win is detected, add this to the list of preferred moves
        - if a loss is detected, add this to the list of moves not to do
    - if no wins or losses are detected, then go one level deeper, and use the
      normal evaluate score scheme of adding scores
"""
# %%
"""
class MetaTicTacToe():
    def __init__(self, size=3, board=None):
        self.size = size
        self.winner = 0  # default is 0 for no winner. 1/2 = O/X. None=draw
        # create the board -- a matrix represents each of the spaces.
        # 0 = empty; 1 = noughts; 2 = crosses
        if board is None:
            self.board = np.zeros((size**2, size**2)).astype(int)
        else:
            self.board = board
        self.metaboard = np.zeros((size, size)).astype(int)

        # create the masks (slice objects) for the grids
        self.grid_masks = [[None for i in range(size)] for j in range(size)]
        for I in range(size):
            for J in range(size):
                mask = np.s_[I*size:I*size+size, J*size:J*size+size]
                self.grid_masks[I][J] = mask

    def ij_to_IJ(self, i, j):
        return i // size, j // size

    def get_grid(self, I, J):1
        mask = self.grid_masks[I][J]
        return self.board[mask]

    def play(self, col, row, value):
        self.board[row, col] = value  # update the value on the board
        self.print_board()  # print the updated board
        I, J = self.ij_to_IJ(col, row)  # grab I,J values of the present grid
        winner = self.check_grid_winner(self.get_grid(I, J))  # check grid
        self.metaboard[I][J] = winner  # update global board
        self.check_global_winner()  # check winner of global board

    def xo(self, c):  # function to convert 0,1 to O,X string
        if c==0:
            return "."
        elif c==1:
            return "O"
        elif c==2:
            return "X"
        else:
            return "?"

    def print_board(self):
        size = self.size
        xo = self.xo
        for ii, row in enumerate(self.board):
            quad = " ".join(["{:<1}",]*size)  # template for numbers in a quad
            line = " | ".join([quad,]*size)  # template for one line of board
            row = [xo(r) for r in row[:]]#map(self.xo, row)
            if (ii % size == 0) and (ii != 0):
                print(line.format(*["-"]*size**2))
            print(line.format(*row))

    def check_grid_winner(self, grid):
        winner = 0  # assume no winner
        rows = [grid[ii, :] for ii in range(self.size)]
        cols = [grid[:, ii] for ii in range(self.size)]
        diags = [np.diag(grid), np.array([grid[-1-ii, ii] for ii in range(3)])]
        lines_to_check = rows + cols + diags

        for l in lines_to_check:
            # if all line elements are not equal, skip this line.
            if not np.all(l == l[0]):
                continue
            elif l[0] == 0:  # if all elements equal zero, skip this line
                continue
            else:
                winner = l[0]
        return winner

    def check_global_winner(self):
        # check the winner of each grid
        for I in range(size):
            for J in range(size):
                grid = self.get_grid(I, J)
                winner = self.check_grid_winner(grid)
                self.metaboard[I][J] = winner

        # check the winner of the meta grid
        WINNER = self.check_grid_winner(self.metaboard)
        if WINNER != 0:
            print("ding ding ding, we have a winner! It's ",WINNER)
            self.winner = WINNER
        else:
            print("no winner yet")

size = 3
game = MetaTicTacToe(size)
for i, j in zip(range(size**2), range(size**2)):
    game.play(i, j, 1)
#"""
# %% new version

class Square():
    def __init__(self, number, winner=None):
        self.winner = winner
        self.number = number

    def __repr__(self):
        return str(self.winner)

    def __str__(self):
        return self.__repr__()

    def assemble_board(self):
        return self.winner


global squareNumber
squareNumber = 0


class metaBoard(Square):
    def __init__(self, size, layers, board=None, current_layer=None):
        self.winner = None
        self.size = size
        self.layers = layers
        self.current_layer = current_layer

        # Do this stuff for the top-level board only
        if self.current_layer is None:
            self.current_layer = 1

            # create the global board (big array without subdivisions)
            # if none was passed
            if board is None:
                globalBoardSize = self.size**layers
                self.globalBoard = np.array([Square(i, i) for i in range(globalBoardSize**2)])
                self.globalBoard = self.globalBoard.reshape(globalBoardSize, globalBoardSize)
            # otherwise use the board passed by the user
            else:
                self.globalBoard = board

            self.board = self.globalBoard

        else:  # if this is not the top-level board
            # assume the parent passed a board
            self.board = board

        # store references to board slices to speed up win checking later
        """for all levels of metaBoard, the slices should be the same. They
        should refer to the slices of the *board*, not the globalBoard"""

        """TODO: fix this. It needs to be slice for the list of sublists. """
        self.rows = [np.s_[ii, :] for ii in range(size)]
        self.cols = [np.s_[:, ii] for ii in range(size)]
        r = list(range(size))
        self.diags = [np.s_[r, r], np.s_[r[::-1], r]]
        self.slices = self.rows + self.cols + self.diags

        # get slices of sub-grids
        # calculate width of subBoard based on current layer
        chunk = int(self.size**self.layers/(2**self.current_layer))
        self.grid_masks = []
        for I in range(self.size):
            for J in range(self.size):
                mask = np.s_[I*chunk:I*chunk+chunk, J*chunk:J*chunk+chunk]
                self.grid_masks.append(mask)

        # if this is not the bottom-level board, create more sub-boards
        self.subBoards = []
        if self.current_layer < self.layers:
            for mask in self.grid_masks:
                args = dict(size=self.size, layers=self.layers,
                            current_layer=self.current_layer+1,
                            board=self.board[mask])
                self.subBoards.append(metaBoard(**args))

        # if this is the bottom-level board, have a list of squares as
        # sub-boards. Because the lowest-level board still needs to be able to
        # ask its sub-boards (squares) what their winners are.
        elif self.current_layer == self.layers:
            # flatten the board to get a list of Square objects
            self.subBoards = list(self.board.flatten())

    def containsSquare(self, squareNumber):
        squareNumbers = [s.number for s in self.board.flatten()]
        return (squareNumber in squareNumbers)

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        return self.__repr__()

    def getNextSubBoard(self, squareNumber):
        """Get the next active subBoard based on the current move (move is
        specified by square number)
        """

        """You need to add some logic for if someone gets sent to a conquered
        subBoard. Or maybe that can be in a getLegalMoves method. This method
        just needs to return the next subBoard. getLegalMoves can then check
        if that subBoard is conquered or not."""

        # if this is not the lowest layer
        if self.current_layer < self.layers:
            # get the subBoard the current move was played in
            moveSubBoard = self.subBoards[self.getMoveSubBoard(squareNumber)]
            # return the index of the subSubBoard in which the move was played
            return moveSubBoard.getMoveSubBoard(squareNumber)
        # prevent this method being called on the lowest layer board.
        else:
            raise Exception("You can't get the next subBoard for the lowest "
                            "layer")

    def getMoveSubBoard(self, squareNumber):
        """Get the index of the subBoard in which the current move is being
        played."""
        for i, subBoard in enumerate(self.subBoards):
            # if this is not the lowest layer, check subBoards
            if self.current_layer < self.layers:
                test = subBoard.containsSquare(squareNumber)
            # if this is the lowest layer, compare to square number directly
            else:
                test = subBoard.number == squareNumber

            if test:
                return i

        raise Exception(f"I couldn't find square number {squareNumber} "
                        "in any of my subBoards. My board looks like:\n"
                        f"{self.board}")

    def play(self, player, squareNumber):
        """IMPLEMENT ME
        This method needs to
        - update the global board
        - report the last played subBoard to the parent board (return number)
        - call the play method of the subBoard in which the Square is (so that
          the subBoard can also log its next active subsubBoard, etc...
        - the top-level metaBoard needs to keep track of whose go it is. Maybe
          have a separate function for this?
        - reuse the k_to_IJ stuff from the previous implementation.
        """

        if not self.containsSquare(squareNumber):
            raise Exception("This board doesn't contain that square")

        # base case
        if self.current_layer == self.layers:
            # find the square that has the right number
            for i, s in enumerate(self.subBoards):
                if s.number == squareNumber:
                    # update the winner of that square to "player" argument
                    s.winner = player
                    return i

        # recursive case
        elif self.current_layer < self.layers:
            # find the subBoard that contains the squareNumber
            for i, s in enumerate(self.subBoards):
                if s.containsSquare(squareNumber):
                    s.play(player, squareNumber)
                    return i

        return "index of the subBoard move was played in"

    def print_board(self):
        print(self.board)

    def plot_board(self, ax=None, depth=1):

        # top layer stuff
        if depth == 1:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 6))
            plt.sca(ax)
            # plt.axis("square")
#            plt.imshow(self.board.astype(str))
            ax.set_xticks([])
            ax.set_yticks([])

        # plot the moves played so far
        for y, row in enumerate(self.board):
            for x, square in enumerate(row):
                ax.text(x, y, square.winner, va="center", ha="center")

        # put recursive stuff here
        # plot the current layer's lines
        width = self.board.shape[0]
        spacing = self.size**(depth-1)
        thickness = 1 + 2*(depth-1)
        locations = np.arange(spacing, width, spacing) - .5
        span = -.5, width-.5
        for loc in locations:
            plt.plot(span, (loc, loc), lw=thickness, c="k")
            plt.plot((loc, loc), span, lw=thickness, c="k")

        if depth < self.layers:
            self.plot_board(ax=ax, depth=depth+1)

        return ax.get_figure(), ax

mb = metaBoard(size=2, layers=2)
mb.print_board()
print("mb.board =\n", mb.board)
fig, ax = mb.plot_board()
mb.play("x", 13)
mb.play("YOMAMA", 0)
fig, ax = mb.plot_board()

