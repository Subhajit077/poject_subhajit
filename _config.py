# Training hyperparameters
batch_size = 64
epochs = 20
learning_rate = 0.001

# Image configuration
resize_x = 32
resize_y = 32
input_channels = 3

# Model configuration
num_classes = 15

# Dataset paths
train_data_path = "data/train"
val_data_path = "data/val"
test_data_path = "data/test"

# Training settings
save_model_path = "_checkpoints/final_weights.pth"