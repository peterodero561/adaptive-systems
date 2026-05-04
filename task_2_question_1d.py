# Multi-Layer Perceptron – learns features in a systematic way
# CCS 2424 Weekend Task – Question One (d)
import numpy as np
# ── Activation functions ───────────────────────────────────────────
def sigmoid(z):       return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z): return sigmoid(z) * (1 - sigmoid(z))
def relu(z):          return np.maximum(0, z)
def relu_deriv(z):    return (z > 0).astype(float)
  
class MLP:
    """
    Multi-Layer Perceptron with arbitrary hidden layers.
    Architecture: Input → [Hidden Layers] → Output
    Training:     Mini-batch stochastic gradient descent + backprop
    """
    def __init__(self, layer_sizes, lr=0.01, epochs=500, batch=32):
        """
        layer_sizes : list of ints, e.g. [784, 128, 64, 10]
        lr          : learning rate (step size for gradient descent)
        epochs      : number of full passes over the training data
        batch       : mini-batch size for stochastic gradient descent
        """
        self.lr, self.epochs, self.batch = lr, epochs, batch
        # He-initialisation: preserves variance through ReLU layers
        self.W = [np.random.randn(layer_sizes[i], layer_sizes[i+1])
                  * np.sqrt(2 / layer_sizes[i])
                  for i in range(len(layer_sizes)-1)]
        self.b = [np.zeros((1, layer_sizes[i+1]))
                  for i in range(len(layer_sizes)-1)]

    # ── Forward pass ────────────────────────────────────────────────
    def forward(self, X):
        """Propagate input X through all layers; cache z & a for backprop."""
        self.z_cache, self.a_cache = [], [X]
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b                                 # linear combination
            a = relu(z) if i < len(self.W)-1 else sigmoid(z)  # activation
            self.z_cache.append(z)
            self.a_cache.append(a)
        return a

    # ── Binary cross-entropy loss ────────────────────────────────────
    def loss(self, y_pred, y_true):
        eps = 1e-9                          # avoid log(0)
        return -np.mean(y_true * np.log(y_pred + eps) +
                        (1 - y_true) * np.log(1 - y_pred + eps))

    # ── Backward pass (backpropagation) ─────────────────────────────
    def backward(self, y_true):
        m   = y_true.shape[0]              # batch size
        dWs, dbs = [], []
        # Output layer gradient
        delta = (self.a_cache[-1] - y_true) / m
        for i in reversed(range(len(self.W))):
            dW = self.a_cache[i].T @ delta  # weight gradient
            db = delta.sum(axis=0, keepdims=True)
            dWs.insert(0, dW)
            dbs.insert(0, db)
            if i > 0:
                # Propagate error back through hidden layer
                delta = (delta @ self.W[i].T) * relu_deriv(self.z_cache[i-1])
        # Gradient descent update
        for i in range(len(self.W)):
            self.W[i] -= self.lr * dWs[i]
            self.b[i] -= self.lr * dbs[i]
          
    # ── Training loop ────────────────────────────────────────────────
    def fit(self, X, y):
        for epoch in range(self.epochs):
            # Shuffle data each epoch to avoid ordering bias
            idx = np.random.permutation(len(X))
            X, y = X[idx], y[idx]
            for start in range(0, len(X), self.batch):
                Xb = X[start:start+self.batch]
                yb = y[start:start+self.batch]
                self.backward(yb) if self.forward(Xb) is not None else None
                self.forward(Xb)  # recalculate after update
                self.backward(yb)
            if epoch % 100 == 0:
                pred = self.forward(X)
                print(f'Epoch {epoch:4d}  Loss={self.loss(pred, y):.4f}')

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

# ── Demo: XOR problem (non-linearly separable) ──────────────────────
if __name__ == '__main__':
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)   # XOR targets
    # Architecture: 2 inputs → 4 hidden → 1 output
    mlp = MLP(layer_sizes=[2, 4, 1], lr=0.1, epochs=500)
    mlp.fit(X, y)
    print('Predictions:', mlp.predict(X).flatten())  # Expected: [0,1,1,0]
