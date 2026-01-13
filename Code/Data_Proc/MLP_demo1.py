import torch
import torch.nn as nn
import torch.optim as optim

class SignalMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SignalMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
    
n_features = 15
n_categories = 4
batch_size = 10

dummy_input = torch.randn(batch_size, n_features)
dummy_labels = torch.randint(0, n_categories, (batch_size,))

model = SignalMLP(n_features, n_categories)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

new_signal = torch.randn(1, n_features)
with torch.no_grad():
    prediction = model(new_signal)
    predicted_class = torch.argmax(prediction, dim=1)
    print(f'Predicted class: {predicted_class.item()}')