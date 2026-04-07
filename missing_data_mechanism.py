import numpy as np
import torch
from scipy import optimize
from sklearn.preprocessing import StandardScaler

##################### MISSING DATA MECHANISMS #############################
## Code directly taken from https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py
##### Missing At Random 

def MAR_mask(X, p, p_obs, rng=None, torch_gen=None):
    """
    Missing at random mechanism with a logistic masking model. First, a subset
    of variables with no missing values is randomly selected. The remaining
    variables have missing values according to a logistic model with random
    weights, re-scaled to attain the desired proportion of missing values.
    """
    n, d = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    to_torch = torch.is_tensor(X)
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    if rng is None:
        rng = np.random.default_rng()
    if torch_gen is None:
        torch_gen = torch.Generator()

    d_obs = max(int(p_obs * d), 1)
    d_na = d - d_obs

    idxs_obs = rng.choice(d, size=d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    coeffs = pick_coeffs(X, idxs_obs, idxs_nas, torch_gen=torch_gen)
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na, generator=torch_gen)
    mask[:, idxs_nas] = ber < ps

    return mask


def MNAR_self_mask_logistic(X, p, rng=None, torch_gen=None):
    """
    Missing not at random mechanism with a logistic self-masking model.
    """
    n, d = X.shape

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    to_torch = torch.is_tensor(X)
    if not to_torch:
        X = torch.from_numpy(X)

    if rng is None:
        rng = np.random.default_rng()
    if torch_gen is None:
        torch_gen = torch.Generator()

    coeffs = pick_coeffs(X, self_mask=True, torch_gen=torch_gen)
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = torch.sigmoid(X * coeffs + intercepts)

    ber = torch.rand(n, d, generator=torch_gen)
    mask = ber < ps

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False, torch_gen=None):
    n, d = X.shape
    if torch_gen is None:
        torch_gen = torch.Generator()
    if self_mask:
        coeffs = torch.randn(d, generator=torch_gen)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na, generator=torch_gen)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


def add_missingness(
    X_noisy,
    noise_level,
    mechanism="MCAR",
    prop_cond_features=0.5,
    rng=None,
    torch_gen=None,
):
    size = X_noisy.shape
    mask = None

    if rng is None:
        rng = np.random.default_rng()
    if torch_gen is None:
        torch_gen = torch.Generator()

    if mechanism == "MCAR":
        mask = rng.random(size) < noise_level

    elif mechanism == "MAR":
        X_float = X_noisy.astype(np.float32).values
        mask = MAR_mask(
            X_float,
            p=noise_level,
            p_obs=prop_cond_features,
            rng=rng,
            torch_gen=torch_gen,
        )

    elif mechanism == "MNAR":
        X_float = X_noisy.astype(np.float32).values
        mask = MNAR_self_mask_logistic(X_float, noise_level, rng=rng, torch_gen=torch_gen)

    if mask is None:
        print("DEBUG : EMPTY MASK")
    return mask
