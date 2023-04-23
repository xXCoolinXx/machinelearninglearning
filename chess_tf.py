import tensorflow as tf
import chess
import chess.pgn
import chess.engine
import numpy as np
import random
import os
import json

# Call stockfish from the command line to play games against a model
# Model has one convolution+pool layer so that it can take in an 8x8 matrix input
# Along with a flatten as well as two dense layers and
# Maps to a position and piece layer matrix that is then translated
# Into standard chess notation which stockfish then accepts as an input
def main():
    #Call chess.engine to play many games against itself 
    #At each point save the board state and the best move predicted by stockfish
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    for i in range(10):
        board = chess.Board()
        while not board.is_game_over():
            #Get the best move from stockfish
            info = engine.analyse(board, chess.engine.Limit(time=0.1))
            board.push(info["move"])
        #Save the board state and the best move by storing it in a json file
        #Being sure to append to the current contents and don't overwrite
        #And use the python json library
        with open("data.json", "a") as f:  
            f.write(json.dumps({"board": board.fen(), "move": info["move"].uci()}) + "\n")

if __name__ == "__main__":
    main()
