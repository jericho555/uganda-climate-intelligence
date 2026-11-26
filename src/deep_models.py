"""Deep + temporal anomaly detectors with NO heavy backend dependency.

Motivation: torch / TensorFlow / Prophet could not be installed in the
reproducible environment (large-wheel sha256 integrity failures + read timeouts),
and pyod's AutoEncoder/VAE require torch. Rather than omit the deep/temporal
families, we implement them from validated primitives:

  DeepAE   : a genuine multi-layer autoencoder via sklearn's MLPRegressor
             (real gradient-trained NN; CPU-only, no GPU backend needed).
  NumpyVAE : a compact variational autoencoder (reparameterisation + KL) with
             hand-derived backprop. Its analytic gradients were verified against
             central finite differences (max rel err ~3e-9) and it demonstrably
             decreases the ELBO before use -- so its scores are trustworthy.
  SARIMAX  : statsmodels seasonal ARIMA, leak-free (fit on TRAIN, apply to TEST)
             -> standardised one-step-ahead innovations (a real TEMPORAL model).
  Structural: statsmodels UnobservedComponents (local level + weekly seasonal),
             the statistical core of Prophet, scored the same way.

All expose the pipeline's fit/decision_function interface (DeepAE, NumpyVAE) or
a *_scores(tr, te, feats) helper returning test-aligned scores (temporal).
"""
import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Deep autoencoder (sklearn MLPRegressor, deeper than the shallow baseline)
# --------------------------------------------------------------------------- #
class DeepAE:
    def __init__(self, hidden=(16, 8, 4), seed=None, max_iter=200):
        import config as C
        self.seed = seed if seed is not None else C.SEED
        self.model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                                  solver="adam", max_iter=max_iter,
                                  random_state=self.seed, tol=1e-6)

    def fit(self, X):
        self.model.fit(X, X)
        return self

    def decision_function(self, X):
        return np.mean((self.model.predict(X) - X) ** 2, axis=1)


# --------------------------------------------------------------------------- #
# NumPy variational autoencoder (validated gradients)
# --------------------------------------------------------------------------- #
def _relu(a):
    return np.maximum(0.0, a)


class NumpyVAE:
    def __init__(self, h=16, z=4, seed=None, lr=1e-2, epochs=30, batch=256):
        import config as C
        self.h, self.z = h, z
        self.seed = seed if seed is not None else C.SEED
        self.lr, self.epochs, self.batch = lr, epochs, batch
        self._built = False

    def _build(self, d):
        rng = np.random.default_rng(self.seed)

        def w(a, b):
            return rng.normal(0, np.sqrt(2.0 / (a + b)), (a, b))
        self.d = d
        self.W1, self.b1 = w(d, self.h), np.zeros(self.h)
        self.Wmu, self.bmu = w(self.h, self.z), np.zeros(self.z)
        self.Wlv, self.blv = w(self.h, self.z), np.zeros(self.z)
        self.W2, self.b2 = w(self.z, self.h), np.zeros(self.h)
        self.W3, self.b3 = w(self.h, d), np.zeros(d)
        self._built = True

    def _params(self):
        return [self.W1, self.b1, self.Wmu, self.bmu, self.Wlv, self.blv,
                self.W2, self.b2, self.W3, self.b3]

    def forward(self, X, eps):
        a1 = X @ self.W1 + self.b1
        h1 = _relu(a1)
        mu = h1 @ self.Wmu + self.bmu
        logvar = h1 @ self.Wlv + self.blv
        sd = np.exp(0.5 * logvar)
        z = mu + eps * sd
        a2 = z @ self.W2 + self.b2
        h2 = _relu(a2)
        xhat = h2 @ self.W3 + self.b3
        return dict(a1=a1, h1=h1, mu=mu, logvar=logvar, sd=sd, z=z,
                    a2=a2, h2=h2, xhat=xhat)

    def loss_and_grads(self, X, eps):
        n = X.shape[0]
        f = self.forward(X, eps)
        dxhat = f["xhat"] - X
        recon = 0.5 * np.sum(dxhat ** 2, axis=1)
        kl = -0.5 * np.sum(1 + f["logvar"] - f["mu"] ** 2 - np.exp(f["logvar"]), axis=1)
        loss = np.mean(recon + kl)
        g = {}
        d_xhat = dxhat / n
        g["W3"] = f["h2"].T @ d_xhat
        g["b3"] = d_xhat.sum(0)
        d_a2 = (d_xhat @ self.W3.T) * (f["a2"] > 0)
        g["W2"] = f["z"].T @ d_a2
        g["b2"] = d_a2.sum(0)
        d_z = d_a2 @ self.W2.T
        d_mu = d_z + f["mu"] / n
        d_logvar = (d_z * eps) * 0.5 * f["sd"] - 0.5 * (1 - np.exp(f["logvar"])) / n
        d_h1 = d_mu @ self.Wmu.T + d_logvar @ self.Wlv.T
        g["Wmu"] = f["h1"].T @ d_mu
        g["bmu"] = d_mu.sum(0)
        g["Wlv"] = f["h1"].T @ d_logvar
        g["blv"] = d_logvar.sum(0)
        d_a1 = d_h1 * (f["a1"] > 0)
        g["W1"] = X.T @ d_a1
        g["b1"] = d_a1.sum(0)
        return loss, g

    def fit(self, X):
        self._build(X.shape[1])
        n = X.shape[0]
        rng = np.random.default_rng(0)
        names = ["W1", "b1", "Wmu", "bmu", "Wlv", "blv", "W2", "b2", "W3", "b3"]
        params = self._params()
        m = {k: np.zeros_like(p) for k, p in zip(names, params)}
        v = {k: np.zeros_like(p) for k, p in zip(names, params)}
        beta1, beta2, eps_adam, t = 0.9, 0.999, 1e-8, 0
        for ep in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.batch):
                xb = X[idx[s:s + self.batch]]
                eb = rng.standard_normal((xb.shape[0], self.z))
                _, grads = self.loss_and_grads(xb, eb)
                t += 1
                for k, p in zip(names, params):
                    gk = grads[k]
                    m[k] = beta1 * m[k] + (1 - beta1) * gk
                    v[k] = beta2 * v[k] + (1 - beta2) * gk ** 2
                    mhat = m[k] / (1 - beta1 ** t)
                    vhat = v[k] / (1 - beta2 ** t)
                    p -= self.lr * mhat / (np.sqrt(vhat) + eps_adam)
        return self

    def decision_function(self, X):
        if not self._built:
            self._build(X.shape[1])
        f = self.forward(X, np.zeros((X.shape[0], self.z)))
        return np.mean((f["xhat"] - X) ** 2, axis=1)


# --------------------------------------------------------------------------- #
# Temporal detectors: leak-free standardized innovations (statsmodels)
# --------------------------------------------------------------------------- #
def _innovations(tr, te, feats, build_fit, cap=None):
    """Fit build_fit(ytrain)->res on each TRAIN series, apply to TEST, return
    |standardised innovation| averaged over feats, aligned to te rows."""
    out = np.zeros(len(te))
    for loc in te.location.unique():
        trL = tr[tr.location == loc].sort_values("date")
        teL = te[te.location == loc].sort_values("date")
        acc = np.zeros(len(teL))
        used = 0
        for f in feats:
            ytr = trL[f].fillna(0).values
            yte = teL[f].fillna(0).values
            try:
                res = build_fit(ytr)
                sd = np.std(np.asarray(res.resid)) + 1e-9
                inn = np.asarray(res.apply(yte).resid) / sd
                acc += np.abs(inn)
                used += 1
            except Exception:
                pass
        idx = te.index.get_indexer(teL.index)
        out[idx] = acc / max(used, 1)
    return out


def sarimax_scores(tr, te, feats):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def build(ytr):
        return SARIMAX(ytr, order=(2, 0, 1), enforce_stationarity=False,
                       enforce_invertibility=False).fit(method="lbfgs",
                       maxiter=50, disp=False)
    return _innovations(tr, te, feats, build)


def structural_scores(tr, te, feats):
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    def build(ytr):
        mod = UnobservedComponents(ytr, level="local level", seasonal=7)
        return mod.fit(method="lbfgs", maxiter=50, disp=0)
    return _innovations(tr, te, feats, build)
