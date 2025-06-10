import torch 
from torch import nn
import random, sys 
import matplotlib.pyplot as plt


class BinaryClassificationModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(), 
            nn.Linear(5, 10),
            nn.ReLU(), 
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def generate_data():
    # dataset 1: x1^2 + x2^2 = y^2
    raw_data = []
    for i in range(500):
        y = 5
        x1 = random.randint(-500, 500)/100
        x2 = (y**2 - x1**2)**0.5
        x1 += random.gauss(0,0.5)
        if random.randint(0,1): x2 *= -1
        x2 += random.gauss(0,0.5)
        raw_data.append([[x1, x2], 0])
    
    for i in range(500):
        y = 10
        x1 = random.randint(-1000, 1000)/100
        x2 = (y**2 - x1**2) ** 0.5
        x1 += random.gauss(0, 0.5)
        if random.randint(0, 1): x2 *= -1
        x2 += random.gauss(0, 0.5)
        raw_data.append([[x1, x2], 1])
    
    random.shuffle(raw_data)
    return raw_data 


def main(epochs, learning_rate):
    raw_data = generate_data()
    X = []
    Y = []
    for data in raw_data:
        X.append(data[0])
        Y.append(data[1])
    
    X_train = torch.tensor(X[:800], dtype=torch.float)
    Y_train = torch.tensor(Y[:800], dtype=torch.float).reshape((-1, 1))
    X_test = torch.tensor(X[800:])
    Y_test = torch.tensor(X[800:]).reshape((-1, 1))

    model = BinaryClassificationModel()
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        
        y_pred = model(X_train)
        loss = loss_fn(y_pred, Y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 100 == 0: print(f"Epoch: {epoch} | Loss: {loss}")
    
    model.eval()
    with torch.inference_mode():
        y_pred = model(X_test)

        pred_labels = (torch.sigmoid(y_pred) > 0.5).int().squeeze().numpy()
        X_test_np = X_test.numpy()
        colors = ['red' if label == 0 else 'blue' for label in pred_labels]
        plt.scatter(X_test_np[:, 0], X_test_np[:, 1], c=colors, alpha=0.6)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()


if __name__ == '__main__':
    try:     
        main(int(sys.argv[1]), float(sys.argv[2]))
    except:
        print("Usage: python binary_classification.py [epochs] [learning_rate]")
        sys.exit()