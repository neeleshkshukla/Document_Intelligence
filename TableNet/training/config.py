import torch

SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0001
EPOCHS = 100
BATCH_SIZE = 2
WEIGHT_DECAY = 3e-4
DATAPATH = '../processed_data_v2.csv'