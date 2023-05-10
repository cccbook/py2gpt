from NoTorch.nn import MLP
from sklearn.datasets import make_moons

"""
Make moons sklearn example, refactored from micrograd/demo.ipynb

"""

# Define network
net = MLP(in_features=2, out_features=1, hidden_sizes=[16, 16, 4])

# Define training data, normalize y
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1


# Train
for step in range(100):
    net.zero_grad()

    out = [net(x) for x in X]

    losses = [(1.0 + -yi * scorei).relu() for yi, scorei in zip(y, out)]

    total_loss = sum(losses)
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y, out)]
    print(f" training accuracy: {(sum(accuracy) / len(accuracy))[0]}")

    total_loss.backward()

    for p in net.parameters():
        p.data -= p.grad * 0.003
