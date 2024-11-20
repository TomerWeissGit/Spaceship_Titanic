import torch
from torch import nn
import numpy as np
from torch import optim

class MainCNN(nn.Module):
    """Simple neural network model for predicting the outcome variable."""
    def __init__(self, input_dim: int, p):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p)
        self.fc2 = nn.Linear(128, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 8)
        self.act3 = nn.ReLU()
        self.dropout2 = nn.Dropout(p)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.dropout1(x)
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)
        return x

    def train_new_data(self, x: np.array, y: np.array, epochs: int,
                       learning_rate: float,
                       early_stopping_patience: int = 100,
                       early_stopping_min_delta: float = 0.0,
                       print_every_x: int = 2) -> None:
        """
        Train the neural network model.
        :param x: np.array, the input data.
        :param y: np.array, the outcome data.
        :param epochs: int, the number of desired epochs.
        :param learning_rate: float, the learning rate for the optimizer.
        :param early_stopping_patience: int, how many epochs to wait before stopping when loss is not improving.
        :param early_stopping_min_delta: float, minimum change in the monitored quantity to qualify as an improvement.
        :param print_every_x: int, print the loss for every x epochs.
        """
        # Convert numpy arrays to torch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Define the loss function and the optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)

        # Training loop
        counter = 0
        for epoch in range(epochs):
            self.train()  # Set the model to training mode

            # Forward pass
            outputs = self(x_tensor)
            loss = criterion(outputs, y_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss for every epoch
            if counter % print_every_x == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}')
            counter += 1
            # check early stopping
            early_stopping(loss.item())
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def predict(self, x_new: np.array) -> np.array:
        """
        Predict the outcome for new input data.
        :param x_new: np.array, the new input data.
        :return: np.array, the predicted outcomes.
        """
        # Convert numpy array to torch tensor
        x_new_tensor = torch.tensor(x_new, dtype=torch.float32)

        # Set the model to evaluation mode
        self.eval()

        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            predictions = self(x_new_tensor)

        # Convert predictions to numpy array and return
        return predictions.numpy()

# helper class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        """
        Early stopping to stop the training when the loss does not improve after certain epochs.
        :param patience: int, how many epochs to wait before stopping when loss is not improving.
        :param min_delta: float, minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
