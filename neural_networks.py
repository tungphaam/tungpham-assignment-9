import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from functools import partial

# Ensure compatibility with non-GUI environments
import matplotlib
matplotlib.use('Agg')

# Create results directory
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define the MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_fn == 'sigmoid':
            sigmoid_x = 1 / (1 + np.exp(-x))
            return sigmoid_x * (1 - sigmoid_x)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))  # Sigmoid for output layer
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        dz2 = self.a2 - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = (dz2 @ self.W2.T) * self.activation_derivative(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.grad_W1 = dW1
        self.grad_W2 = dW2

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int).reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    # Perform more training steps per frame to speed up convergence
    for _ in range(50):  # Increased from 10 to 50 steps per frame
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden Space Plot
    ax_hidden.clear()
    hidden_features = mlp.a1
    hidden_features = (hidden_features - hidden_features.mean(axis=0)) / (hidden_features.std(axis=0) + 1e-8)
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)

    # Decision Hyperplane in Hidden Space
    x_range = np.linspace(-1.5, 1.5, 20)
    y_range = np.linspace(-1.5, 1.5, 20)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = -(mlp.W2[0, 0] * xx + mlp.W2[1, 0] * yy + mlp.b2[0, 0]) / (mlp.W2[2, 0] + 1e-8)
    ax_hidden.plot_surface(xx, yy, zz, alpha=0.3, color='yellow')
    ax_hidden.set_xlim([-1.5, 1.5])
    ax_hidden.set_ylim([-1.5, 1.5])
    ax_hidden.set_zlim([-1.5, 1.5])
    ax_hidden.set_title(f"Hidden Space at Step {frame * 50}")
    ax_hidden.set_xlabel("Hidden Unit 1")
    ax_hidden.set_ylabel("Hidden Unit 2")
    ax_hidden.set_zlabel("Hidden Unit 3")

    # Input Space Decision Boundary
    ax_input.clear()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap='bwr', alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k', s=30)
    ax_input.set_xlim([x_min, x_max])
    ax_input.set_ylim([y_min, y_max])
    ax_input.set_title(f"Input Space at Step {frame * 50}")
    ax_input.set_xlabel("X1")
    ax_input.set_ylabel("X2")

    # Gradients Visualization
    ax_gradient.clear()
    nodes_input = ['x1', 'x2']
    nodes_hidden = [f"h{i+1}" for i in range(mlp.W1.shape[1])]
    nodes_output = ['y']
    for i, label in enumerate(nodes_input):
        ax_gradient.scatter(i, 0, s=500, color='blue', alpha=0.8)
        ax_gradient.text(i, -0.1, label, fontsize=12, ha='center')
    for i, label in enumerate(nodes_hidden):
        ax_gradient.scatter(i + len(nodes_input), 1, s=500, color='blue', alpha=0.8)
        ax_gradient.text(i + len(nodes_input), 1.1, label, fontsize=12, ha='center')
    for i, label in enumerate(nodes_output):
        ax_gradient.scatter(i + len(nodes_input) + len(nodes_hidden), 2, s=500, color='blue', alpha=0.8)
        ax_gradient.text(i + len(nodes_input) + len(nodes_hidden), 2.1, label, fontsize=12, ha='center')
    for i in range(mlp.W1.shape[0]):
        for j in range(mlp.W1.shape[1]):
            weight_magnitude = np.abs(mlp.grad_W1[i, j]) / (np.max(np.abs(mlp.grad_W1)) + 1e-8)
            ax_gradient.plot([i, j + len(nodes_input)], [0, 1], 'purple', alpha=weight_magnitude, linewidth=3 * weight_magnitude)
    for j in range(mlp.W2.shape[0]):
        for k in range(mlp.W2.shape[1]):
            weight_magnitude = np.abs(mlp.grad_W2[j, k]) / (np.max(np.abs(mlp.grad_W2)) + 1e-8)
            ax_gradient.plot([j + len(nodes_input), k + len(nodes_input) + len(nodes_hidden)], [1, 2], 'red', alpha=weight_magnitude, linewidth=3 * weight_magnitude)
    ax_gradient.set_title(f"Gradients at Step {frame * 50}")
    ax_gradient.axis("off")

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)
    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
        frames=int(step_num / 10),
        interval=200,  # Add a 200 ms delay per frame
        repeat=False
    )
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)