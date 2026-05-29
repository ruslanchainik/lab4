import numpy as np


def brelu(x):
    return np.clip(x, 0.0, 1.0)


def brelu_grad(x):
    return ((x > 0.0) & (x < 1.0)).astype(x.dtype)


def sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def bce_loss(y_true, y_pred, eps=1e-12):
    p = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


class Dense:
    def __init__(self, n_in, n_out, rng):
        scale = np.sqrt(2.0 / n_in)
        self.W = rng.standard_normal((n_in, n_out)) * scale
        self.b = np.zeros((1, n_out))
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.dW = self.x.T @ grad_out / self.x.shape[0]
        self.db = grad_out.mean(axis=0, keepdims=True)
        return grad_out @ self.W.T

    def params(self):
        return [(self.W, lambda v: setattr(self, "W", v), self.dW),
                (self.b, lambda v: setattr(self, "b", v), self.db)]


class SingleLayerNet:
    def __init__(self, n_in, rng):
        self.fc = Dense(n_in, 1, rng)

    def forward(self, x):
        z = self.fc.forward(x)
        self._mask = brelu_grad(z)
        a = brelu(z)
        return sigmoid(a)

    def backward(self, y_true, y_pred):
        grad_a = (y_pred - y_true)
        grad_z = grad_a * self._mask
        self.fc.backward(grad_z)

    def layers(self):
        return [self.fc]

    def predict_proba(self, x):
        return self.forward(x)


class Perceptron:
    def __init__(self, n_in, n_hidden, rng):
        self.fc1 = Dense(n_in, n_hidden, rng)
        self.fc2 = Dense(n_hidden, 1, rng)

    def forward(self, x):
        z1 = self.fc1.forward(x)
        self._mask = brelu_grad(z1)
        h = brelu(z1)
        z2 = self.fc2.forward(h)
        return sigmoid(z2)

    def backward(self, y_true, y_pred):
        grad_z2 = (y_pred - y_true)
        grad_h = self.fc2.backward(grad_z2)
        grad_z1 = grad_h * self._mask
        self.fc1.backward(grad_z1)

    def layers(self):
        return [self.fc1, self.fc2]

    def predict_proba(self, x):
        return self.forward(x)


class NadamOptimizer:
    def __init__(self, layers, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.t = 0
        self.state = []
        for layer in layers:
            for p, _, _ in layer.params():
                self.state.append({"m": np.zeros_like(p), "v": np.zeros_like(p)})
        self.layers = layers

    def step(self):
        self.t += 1
        idx = 0
        for layer in self.layers:
            for p, setter, g in layer.params():
                st = self.state[idx]
                idx += 1
                st["m"] = self.b1 * st["m"] + (1.0 - self.b1) * g
                st["v"] = self.b2 * st["v"] + (1.0 - self.b2) * (g * g)
                m_hat = st["m"] / (1.0 - self.b1 ** self.t)
                v_hat = st["v"] / (1.0 - self.b2 ** self.t)
                nesterov = self.b1 * m_hat + (1.0 - self.b1) * g / (1.0 - self.b1 ** self.t)
                setter(p - self.lr * nesterov / (np.sqrt(v_hat) + self.eps))


def iterate_minibatches(X, y, batch_size, rng):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


def train(model, X_tr, y_tr, X_val, y_val, lr=1e-2, epochs=300,
          batch_size=32, seed=0, patience=50):
    rng = np.random.default_rng(seed)
    opt = NadamOptimizer(model.layers(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = np.inf
    best_state = None
    bad = 0
    for ep in range(epochs):
        for xb, yb in iterate_minibatches(X_tr, y_tr, batch_size, rng):
            p = model.forward(xb)
            model.backward(yb, p)
            opt.step()
        p_tr = model.forward(X_tr)
        p_val = model.forward(X_val)
        tl = bce_loss(y_tr, p_tr)
        vl = bce_loss(y_val, p_val)
        ta = ((p_tr >= 0.5) == y_tr).mean()
        va = ((p_val >= 0.5) == y_val).mean()
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        if vl < best_val - 1e-5:
            best_val = vl
            best_state = [(layer.W.copy(), layer.b.copy()) for layer in model.layers()]
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        for layer, (W, b) in zip(model.layers(), best_state):
            layer.W = W
            layer.b = b
    return history
