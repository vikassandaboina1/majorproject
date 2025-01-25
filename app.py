import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and preprocess the Iris dataset
iris = load_iris()
data = iris['data']
target = iris['target']

# Standardize the features
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Create a similarity-based graph (simple pairwise Euclidean distance)
from sklearn.metrics.pairwise import pairwise_distances
distances = pairwise_distances(data)
threshold = 0.7  # Similarity threshold for creating edges

# Create edges based on distance threshold
edges = np.where(distances < threshold)

# Create edge index
edge_index = torch.tensor(edges, dtype=torch.long)

# Create node features
x = torch.tensor(data, dtype=torch.float)

# Labels for classification (species of the iris flower)
y = torch.tensor(target, dtype=torch.long)

# Temporal information (simulating time steps)
# Here we just use a fake time sequence for simplicity (not actual time-based data)
# In real cases, you'd have time steps and changes in graph structure.
time_steps = 3  # Simulate 3 time steps

# Prepare Data object for PyTorch Geometric
# Each time step could have a different graph, here we keep it simple with the same graph for all steps
data_list = []
for t in range(time_steps):
    data_list.append(Data(x=x, edge_index=edge_index, y=y))

# Simple Temporal Graph Network (GCN based)
class TGN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TGN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, data_list):
        x_list = []
        for data in data_list:
            # Apply graph convolutions for each time step's graph
            x = self.conv1(data.x, data.edge_index)
            x = F.relu(x)
            x = self.conv2(x, data.edge_index)
            x_list.append(x)
        
        # Pooling or aggregation step (we take the final output of the last time step)
        return x_list[-1]

# Initialize model, optimizer, and loss function
model = TGN(in_channels=4, out_channels=3)  # 4 input features, 3 classes (species)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data_list)
    
    # Compute loss
    loss = criterion(out, y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

# After training, evaluate the model
model.eval()
out = model(data_list)
_, pred = out.max(dim=1)
correct = (pred == y).sum().item()
accuracy = correct / y.size(0)
print(f"Accuracy: {accuracy * 100:.2f}%")
