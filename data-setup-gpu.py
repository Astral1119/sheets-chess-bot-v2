# dependencies
'''
%pip install requests
%pip install bitarray
%pip install peewee
%pip install tqdm
'''

import json
import os
import requests
import base64
from peewee import *
from tqdm import tqdm
import cupy as cp
import numpy as np
import math


# uses lichess's evals db
# https://database.lichess.org/lichess_db_eval.jsonl.zst

# if db is already downloaded, use lichess_db_eval.jsonl
# if not, download it
if not os.path.exists('lichess_db_eval.jsonl'):
    url = 'https://database.lichess.org/lichess_db_eval.jsonl.zst'
    r = requests.get(url)
    with open('lichess_db_eval.jsonl.zst', 'wb') as f:
        f.write(r.content)
    os.system('zstd -d lichess_db_eval.jsonl.zst')


# we can encode the fens as bitmaps
# p, n, b, r, q, k is 6 types of piece
# w, b is 2 colors
# 6 * 2 * 8 * 8 = 768
# one bit for whether it's white's turn
# castling is four bits for KQkq
# reserve seven bits for en passant, with 0 being no en passant and 1-64 for the square
# 768 + 1 + 4 + 7 = 780

def fen_to_bitmap(fen):
    """
    Converts a FEN string to a GPU-backed bitmap representation using CuPy.
    """
    # Start by splitting the FEN string into its parts
    parts = fen.split(' ')
    board = parts[0]
    to_move = parts[1]
    castling = parts[2]
    ep = parts[3]

    # Create an empty CuPy array (similar to the size of the original bitarray)
    bitmap = cp.zeros(780, dtype=cp.bool_)

    # Fill in the board
    row = 0
    col = 0

    # Char order for pieces
    pieces = 'PNBRQKpnbrqk'

    for char in board:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            piece = pieces.index(char)
            bitmap[piece * 64 + row * 8 + col] = 1
            col += 1

    # Fill in "to move"
    bitmap[768] = (to_move == 'w')

    # Fill in castling rights
    bitmap[769] = ('K' in castling)
    bitmap[770] = ('Q' in castling)
    bitmap[771] = ('k' in castling)
    bitmap[772] = ('q' in castling)

    # Fill in en passant square, if applicable
    if ep != '-':
        col = ord(ep[0]) - ord('a')  # Convert file letter to column index
        row = int(ep[1]) - 1  # Convert rank number to row index
        square = row * 8 + col  # Calculate square index

        # Create a 7-bit CuPy array for en passant
        ep_array = cp.zeros(7, dtype=cp.bool_)
        ep_array[0] = 1  # First bit indicates en passant is possible

        # Six bits to encode the square index
        for i in range(6):
            ep_array[i + 1] = square % 2
            square //= 2

        # Place the en passant array into the bitmap
        bitmap[773:780] = ep_array

    return np.packbits(bitmap)

# cleaning the db
# we only want the eval with the most knodes
# and we want to convert the fens to bitmaps
# cleaning the db with optimizations
# for mate in x, we want to convert x to a cp of 1000000 / x
# batch processing of FEN positions, minimizing GPU data transfers, and batch insert into SQLite

# SQLite database setup

sqlite_db = SqliteDatabase('lichess.db')


# Database model
class Evaluations(Model):
    id = IntegerField(primary_key=True)
    fen = TextField()
    binary = BlobField()  # Blob for the bitmap
    eval = FloatField()

    class Meta:
        database = sqlite_db

    def binary_base64(self):
        return base64.b64encode(self.binary)


# Initialize the database table
sqlite_db.connect()
sqlite_db.create_tables([Evaluations])

# Parameters
# num_positions = 190987505
num_positions = 10000000
batch_size = 50000  # Process positions in batches
position_id = 1

# Define SQLite's maximum variable limit (adjust if your environment's limit is known to differ)
SQLITE_MAX_VARIABLES = 999


# Calculate the maximum allowable batch size based on the number of columns in the table
def get_max_batch_size(column_count):
    return math.floor(SQLITE_MAX_VARIABLES / column_count)


# Calculate the maximum batch size for the Evaluations table
column_count = len(Evaluations._meta.sorted_fields)  # Total columns in the Evaluations table
max_batch_size = get_max_batch_size(column_count)


# Process positions from the JSONL file
with open('lichess_db_eval.jsonl', 'r') as f:
    rows = []  # Buffer for batched database rows
    for _ in tqdm(range(num_positions)):
        line = f.readline()
        if not line:  # End of file
            break

        # Parse the JSONL line
        position = json.loads(line)
        best_eval = max(position['evals'], key=lambda x: x['knodes'])

        # GPU processing: Convert FEN to bitmap
        bitmap = fen_to_bitmap(position['fen'])

        # Evaluation conversion
        if 'mate' in best_eval['pvs'][0]:
            eval = 1000000 / best_eval['pvs'][0]['mate']
        else:
            eval = best_eval['pvs'][0]['cp']

        # Prepare data for batch insertion
        rows.append({
            "fen": position['fen'],
            "binary": cp.asnumpy(bitmap).tobytes(),  # Transfer bitmap to CPU and serialize
            "eval": eval
        })

        # Batch insert into the database
        if len(rows) >= max_batch_size:  # Use dynamically-calculated max batch size
            Evaluations.insert_many(rows).execute()
            rows = []  # Clear the batch buffer

    # Insert any remaining rows after loop
    if rows:
        Evaluations.insert_many(rows).execute()

# testing the db
# get the first 10 entries
for eval in Evaluations.select().limit(10):
    print(eval.fen, eval.eval)
    print(eval.binary_base64())

sqlite_db.close()
