import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# =========================
# 1. Data loading
# =========================
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# =========================
# 2. Neural Network Model
# =========================
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Flatten(),              # 28x28 → 784
            nn.Linear(784, 128),       # hidden layer
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)          # output (0~9)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleNN()

# =========================
# 3. Loss + Optimizer
# =========================
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 4. Training loop
# =========================
epochs = 3

for epoch in range(epochs):
    for images, labels in train_loader:

        # forward
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# =========================
# 5. Evaluation
# =========================
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy:", correct / total)