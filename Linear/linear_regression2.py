import torch 
from torch import nn 
import random, sys
import matplotlib.pyplot as plt

class SimpleLinearModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.model(x)

def main(desired_gradient, desired_intercept, epochs, learning_rate):
    X_list = torch.arange(0, 100, 1, dtype=torch.float)
    errors = torch.tensor([random.gauss(0,0.5) for _ in range(100)])
    Y_list = desired_gradient*X_list + desired_intercept
    Y_list = Y_list + errors

    X_train = X_list[:80].unsqueeze(1)
    Y_train = Y_list[:80].unsqueeze(1)
    model = SimpleLinearModel()
    loss_fn = nn.L1Loss()
    optimiser = torch.optim.SGD(params=list(model.parameters()), lr=learning_rate)  

    for epoch in range(int(epochs)):
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, Y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if epoch % 1000 == 0:
            print(f"epoch: {epoch} | loss: {loss}")
    
    print(list(model.parameters()))
    # Plotting predictions vs actual values
    model.eval()
    with torch.no_grad():
        X_test = X_list[80:].unsqueeze(1)
        Y_test = Y_list[80:].unsqueeze(1)
        Y_pred = model(X_test).squeeze()

    plt.figure(figsize=(8, 5))
    plt.scatter(X_test.squeeze(), Y_test.squeeze(), label='Actual', color='blue')
    plt.plot(X_test.squeeze(), Y_pred, label='Predicted', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Model Accuracy: Actual vs Predicted')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python linear_regression.py [gradient] [intercept] [epochs] [learning rate]")
        sys.exit()
    
    args = list(map(float, sys.argv[1:]))
    main(args[0], args[1], args[2], args[3])