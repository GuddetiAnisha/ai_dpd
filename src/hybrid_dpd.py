import numpy as np
import torch
from torch import nn

def _delay(x, m):
    if m == 0:
        return x
    return np.concatenate([np.zeros(m, dtype=x.dtype), x[:-m]])

def build_features(x, baseline, memory_depth=4):
    feats = []
    for m in range(memory_depth):
        xm = _delay(x, m)
        feats += [xm.real, xm.imag, np.abs(xm)]
    feats += [baseline.real, baseline.imag, np.abs(baseline)]
    return np.column_stack(feats).astype(np.float32)

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden=(48,48)):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class HybridResidualDPD:
    def __init__(self, baseline_model, memory_depth=4, hidden=(48,48), lr=8e-4,
                 epochs=10, batch_size=1024, seed=7, max_train_samples=35000,
                 residual_scale=0.35):
        self.baseline_model = baseline_model
        self.memory_depth = memory_depth
        self.hidden = tuple(hidden)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.max_train_samples = max_train_samples
        self.residual_scale = residual_scale
        self.model = None
        self.mu = None
        self.sigma = None

    def fit(self, pa_output, desired_input):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        baseline = self.baseline_model.predict(pa_output)
        X = build_features(pa_output, baseline, self.memory_depth)
        residual = desired_input - baseline
        y = np.column_stack([residual.real, residual.imag]).astype(np.float32)

        if len(X) > self.max_train_samples:
            idx = np.linspace(0, len(X)-1, self.max_train_samples).astype(int)
            X, y = X[idx], y[idx]

        self.mu = X.mean(axis=0, keepdims=True)
        self.sigma = X.std(axis=0, keepdims=True) + 1e-6
        X = (X - self.mu)/self.sigma

        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model = ResidualMLP(X.shape[1], self.hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict(self, x):
        if self.model is None:
            raise RuntimeError("Hybrid model not fitted")
        baseline = self.baseline_model.predict(x)
        X = build_features(x, baseline, self.memory_depth)
        X = (X - self.mu)/self.sigma
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.from_numpy(X)).numpy()
        residual = pred[:,0] + 1j*pred[:,1]
        return baseline + self.residual_scale*residual

    @property
    def parameter_count(self):
        if self.model is None:
            return None
        return self.baseline_model.parameter_count + sum(p.numel() for p in self.model.parameters())

    def approx_memory_bytes(self):
        if self.model is None:
            return None
        neural = sum(p.numel() for p in self.model.parameters()) * 4
        return neural + self.baseline_model.approx_memory_bytes()

    def approx_macs_per_sample(self):
        if self.model is None:
            return None
        dense = 0
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                dense += m.in_features*m.out_features
        return dense + self.baseline_model.approx_macs_per_sample()
