import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from model.tgn import TGN
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from utils.data_processing import compute_time_statistics, get_data_node_classification
from evaluation.evaluation import eval_node_classification

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='How often to backpropagate')
parser.add_argument('--use_memory', action='store_true', help='Use memory augmentation')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true', help='Update memory at batch end')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for each user')
parser.add_argument('--different_new_nodes', action='store_true', help='Use disjoint new nodes for train/val')
parser.add_argument('--uniform', action='store_true', help='Uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true', help='Randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true', help='Use destination embedding in message')
parser.add_argument('--use_source_embedding_in_message', action='store_true', help='Use source embedding in message')
parser.add_argument('--dyrep', action='store_true', help='Use dyrep for message calculation')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
DATA = args.data
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

# Create necessary directories
Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}-self-supervised.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}-self-supervised.pth'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

# Function to load data with absolute path
def get_data(dataset_name):
    file_path = f'C:/Users/vikky/Downloads/tgn-master (2)/tgn-master/data/ml_wikipedia.npy'
    logger.info(f"Loading dataset from {file_path}")
    return pd.read_csv(file_path)

# Load data
full_data, node_features, edge_features, train_data, val_data, test_data = get_data_node_classification(DATA)

# Initialize the TGN model
device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
model = TGN(node_features.shape[1], edge_features.shape[1], args.memory_dim, args.message_dim, args.time_dim, args.node_dim,
            args.n_degree, args.n_head, args.n_layer, args.embedding_module, args.message_function, args.aggregator,
            args.memory_update_at_end, args.use_source_embedding_in_message, args.use_destination_embedding_in_message,
            args.dyrep, dropout=args.drop_out, device=device)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

early_stopper = EarlyStopMonitor(max_round=args.patience)

logger.info("Starting training...")
for epoch in range(NUM_EPOCH):
    model.train()
    losses = []
    for batch_idx, batch in enumerate(train_data.get_batches(BATCH_SIZE)):
        optimizer.zero_grad()
        # Forward pass
        predictions, _ = model(batch.src, batch.dst, batch.t, batch.label)
        loss = criterion(predictions, batch.label)
        losses.append(loss.item())
        loss.backward()

        # Update weights
        optimizer.step()

    avg_loss = np.mean(losses)
    logger.info(f"Epoch {epoch}: Loss = {avg_loss}")

    # Validation step
    model.eval()
    with torch.no_grad():
        val_loss = []
        for batch in val_data.get_batches(BATCH_SIZE):
            predictions, _ = model(batch.src, batch.dst, batch.t, batch.label)
            val_loss.append(criterion(predictions, batch.label).item())

        val_avg_loss = np.mean(val_loss)
        logger.info(f"Validation Loss = {val_avg_loss}")

        # Early stopping
        if early_stopper.step(val_avg_loss):
            logger.info("Early stopping triggered.")
            break

logger.info("Training completed!")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
logger.info(f"Model saved to {MODEL_SAVE_PATH}")
