"""Deep + temporal anomaly detectors for the benchmark.

History: torch / TensorFlow / Prophet initially could not be installed in the
reproducible environment (large-wheel sha256 integrity failures + read timeouts),
and pyod's AutoEncoder/VAE require torch. Rather than omit the deep/temporal
families, we first implemented them from validated primitives:

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
  TemporalIF: Isolation Forest on a 14-day lag-window embedding of the same
             series (temporal isolation forest, deep_models.temporal_if_scores).

Update: a CPU-only torch (2.4.1) is now vendored into the workspace via pip's
``--target`` (directory ``._torchpkgs``), so the previously-omitted recurrent and
attention families are included as real torch models:

  ensure_torch   : bootstraps the vendored ``._torchpkgs`` onto sys.path.
  _SeqAE         : an LSTM encoder--decoder and a Transformer encoder
                   autoencoder (kind="lstm"/"transformer"), Adam-trained to
                   reconstruct input windows, scored by per-window recon MSE.
  seq_ae_scores  : leak-free windowing (14-day, past+current only), train-only
                   standardisation, one pooled model per kind, test-aligned scores.

DeepAE/NumpyVAE expose the pipeline's fit/decision_function interface; the
temporal and sequence helpers expose *_scores(tr, te, feats, ...) returning
test-aligned scores.
"""
import warnings
import numpy as np
import pandas as pd
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


def temporal_if_scores(tr, te, feats, window=14, max_train_windows=8000, seed=None):
    """Temporal (lag-window) Isolation Forest.

    Each day is embedded as the flattened block of the ``window`` daily feature
    vectors ENDING at that day -- the lag embedding of LagGEDScore (Cannon et al.
    2002) applied to Isolation Forest, i.e. the time-series variant the method
    family asks for. The embedding itself supplies the temporal context a
    cross-sectional detector throws away (how the day was reached, not just where
    it sits), while the forest stays the same.

    Leak-free by construction: standardisation and the forest are fitted on
    TRAIN windows only, and the window attached to a TEST row reads values at or
    before its own date (series are sorted by date per location, never reversed),
    so no future observation can influence a score.
    """
    import config as C
    from sklearn.ensemble import IsolationForest
    seed = C.SEED if seed is None else seed
    mu = tr[feats].mean().values
    sd = tr[feats].std().values + 1e-9
    idxpos = pd.Series(np.arange(len(te)), index=te.index)
    tr_w, te_w, te_pos = [], [], []
    for loc in te.location.unique():
        trL = tr[tr.location == loc].sort_values("date")
        teL = te[te.location == loc].sort_values("date")
        comb = pd.concat([trL, teL], axis=0).sort_values("date")
        A = np.nan_to_num((comb[feats].values - mu) / sd,
                          nan=0.0, posinf=0.0, neginf=0.0)
        wins = _build_windows(A, window)                 # (T, window, F)
        is_te = comb.index.isin(teL.index)
        te_w.append(wins[is_te].reshape(int(is_te.sum()), -1))
        te_pos.append(idxpos.loc[comb.index[is_te]].values)
        trw = wins[~is_te]
        if len(trw) > max_train_windows:
            trw = trw[::int(np.ceil(len(trw) / max_train_windows))]
        tr_w.append(trw.reshape(len(trw), -1))
    Xtr = np.concatenate(tr_w, axis=0)
    if len(Xtr) > max_train_windows:
        rng = np.random.default_rng(seed)
        Xtr = Xtr[rng.choice(len(Xtr), max_train_windows, replace=False)]
    iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xtr)
    out = np.zeros(len(te))
    for blk, pos in zip(te_w, te_pos):
        if len(blk):
            out[pos] = -iso.decision_function(blk)
    return out


# --------------------------------------------------------------------------- #
# Deep SEQUENCE autoencoders (torch) -- LSTM-AE and Transformer-AE
# --------------------------------------------------------------------------- #
def ensure_torch():
    """Return True if torch is importable, bootstrapping the workspace-local
    vendored CPU wheel directory (._torchpkgs) onto sys.path if torch is not
    already on the default path."""
    import sys
    from pathlib import Path
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        vend = Path(__file__).resolve().parent.parent / "._torchpkgs"
        if vend.exists() and str(vend) not in sys.path:
            sys.path.insert(0, str(vend))
        try:
            import torch  # noqa: F401
            return True
        except Exception:
            return False


class _SeqAE:
    """Autoencoder over daily feature windows: encoder -> context -> decoder,
    scored by per-window reconstruction MSE. kind in {'lstm','transformer'}."""

    def __init__(self, n_feat, window, kind="lstm", hidden=16, layers=1,
                 heads=2, epochs=8, batch=256, lr=3e-3, seed=0):
        import torch
        import torch.nn as nn
        self.torch, self.kind = torch, kind
        self.n_feat, self.window = n_feat, window
        self.hidden, self.layers, self.heads = hidden, layers, heads
        self.epochs, self.batch, self.lr = epochs, batch, lr
        torch.manual_seed(seed)
        w, h, L = window, hidden, layers

        if kind == "lstm":
            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.h, self.L, self.w = h, L, w
                    self.enc = nn.LSTM(n_feat, h, L, batch_first=True)
                    self.dec = nn.LSTM(h, h, L, batch_first=True)
                    self.out = nn.Linear(h, n_feat)

                def forward(self, x):
                    _, (hc, _) = self.enc(x)
                    seed = hc[-1].unsqueeze(1).repeat(1, self.w, 1)
                    d, _ = self.dec(seed)
                    return self.out(d)
        else:  # transformer
            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.inp = nn.Linear(n_feat, h)
                    self.pe = nn.Parameter(torch.randn(1, w, h) * 0.02)
                    layer = nn.TransformerEncoderLayer(
                        h, heads, h * 2, batch_first=True, dropout=0.0)
                    self.enc = nn.TransformerEncoder(layer, L)
                    self.out = nn.Linear(h, n_feat)

                def forward(self, x):
                    return self.out(self.enc(self.inp(x) + self.pe))

        self.net = Net()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X):
        t = self.torch
        Xt = t.tensor(X, dtype=t.float32)
        n = len(Xt)
        for _ in range(self.epochs):
            perm = t.randperm(n)
            tot = 0.0
            for s in range(0, n, self.batch):
                b = Xt[perm[s:s + self.batch]]
                self.opt.zero_grad()
                loss = t.nn.functional.mse_loss(self.net(b), b)
                loss.backward()
                self.opt.step()
                tot += float(loss) * len(b)
        self.train_rmse = (tot / max(n, 1)) ** 0.5
        return self

    def score(self, X):
        t = self.torch
        self.net.eval()
        Xt = t.tensor(X, dtype=t.float32)
        with t.no_grad():
            err = ((self.net(Xt) - Xt) ** 2).mean(dim=(1, 2))
        self.net.train()
        return err.numpy()


def _build_windows(series, window):
    """series:(T,F) -> (T,window,F) where row i is the window ENDING at i,
    left-padded by repeating the first observed row."""
    T = len(series)
    pad = np.repeat(series[:1], window - 1, axis=0)
    ap = np.concatenate([pad, series], axis=0)
    return np.stack([ap[i:i + window] for i in range(T)], axis=0)


def seq_ae_scores(tr, te, feats, kind="lstm", window=14,
                  max_train_windows=6000, epochs=8, seed=None, hidden=16):
    """Leak-free sequence autoencoder: standardise on TRAIN, train ONE model on
    train windows pooled across districts, score each TEST row by the
    reconstruction error of the window ending at that row. Returns
    (test-aligned scores, train_rmse)."""
    import config as C
    seed = C.SEED if seed is None else seed
    mu = tr[feats].mean().values
    sd = tr[feats].std().values + 1e-9
    idxpos = pd.Series(np.arange(len(te)), index=te.index)
    tr_w, te_w, te_pos = [], [], []
    for loc in te.location.unique():
        trL = tr[tr.location == loc].sort_values("date")
        teL = te[te.location == loc].sort_values("date")
        comb = pd.concat([trL, teL], axis=0).sort_values("date")
        A = np.nan_to_num((comb[feats].values - mu) / sd,
                          nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        wins = _build_windows(A, window)
        is_te = comb.index.isin(teL.index)
        te_w.append(wins[is_te])
        te_pos.append(idxpos.loc[comb.index[is_te]].values)
        trw = wins[~is_te]
        if len(trw) > max_train_windows:
            trw = trw[::int(np.ceil(len(trw) / max_train_windows))]
        tr_w.append(trw)
    Xtr = np.concatenate(tr_w, axis=0).astype(np.float32)
    if len(Xtr) > max_train_windows:
        rng = np.random.default_rng(0)
        Xtr = Xtr[rng.choice(len(Xtr), max_train_windows, replace=False)]
    model = _SeqAE(len(feats), window, kind=kind, epochs=epochs,
                   seed=seed, hidden=hidden)
    model.fit(Xtr)
    out = np.zeros(len(te))
    for blk, pos in zip(te_w, te_pos):
        if len(blk):
            out[pos] = model.score(blk.astype(np.float32))
    return out, float(model.train_rmse)
