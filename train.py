import torch
import deepspeed
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Dummy dataset
class CustomDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Training function with DeepSpeed
def train():
    model = SimpleModel()

    # Initialize DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config.json"
    )

    # Load dataset
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):  # Train for 5 epochs
        for data, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            model.backward(loss)
            optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss = {loss.item()}")

    # Save trained model
    torch.save(model.state_dict(), "trained_model.pth")

if __name__ == "__main__":
    train()
