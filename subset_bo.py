import math
import torch
import gpytorch
import botorch
from gpytorch.constraints.constraints import Positive, Interval
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.models import SingleTaskGP
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.utils.gpytorch_modules import (
    get_matern_kernel_with_gamma_prior,
    get_covar_module_with_dim_scaled_prior,
)
from typing import Optional


class SumKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, sub_kernel: Optional[gpytorch.kernels.Kernel] = None, **kwargs):
        super().__init__(**kwargs)
        if sub_kernel is None:
            sub_kernel = get_matern_kernel_with_gamma_prior(ard_num_dims=1)
        self.sub_kernel = sub_kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor, **params):
        xsum = x.sum(dim=-1, keepdim=True)
        ysum = y.sum(dim=-1, keepdim=True)
        ksize = self.sub_kernel(xsum, ysum, **params)
        return ksize


class DoubleSumKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(
        self,
        features: torch.Tensor,
        inner_kernel: Optional[gpytorch.kernels.Kernel] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.features = features
        if inner_kernel is None:
            inner_kernel = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=self.features.shape[-1]
            )
        self.inner_kernel = inner_kernel

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params
    ) -> torch.Tensor:
        r"""
        Compute the covariance matrix between x1 and x2.

        Parameters
        ----------
            x1: torch.Tensor
                First set of data (... x N x D)
                0/1-valued tensor.
            x2 torch.Tensor
                Second set of data (... x M x D)
                0/1-valued tensor.
            diag: bool
                Should the Kernel compute the whole kernel, or just the diag?
                If True, it must be the case that `x1 == x2`. (Default: False.)

        Returns
        -------
            The kernel matrix or vector. The shape depends on the kernel's evaluation mode:

            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x K x N`

        """
        assert x1.shape[-1] == len(self.features)

        d = len(self.features)
        F = self.inner_kernel.forward(self.features, self.features)
        assert F.shape == torch.Size([d, d])

        if diag:
            assert x1.shape == x2.shape
            K = torch.einsum("...np,...nq,pq->...n", x1, x2, F)
            D = (x1.sum(dim=-1) ** 2).clamp(min=1)
            assert K.shape == D.shape
            K /= D
            return K
        else:
            K = torch.einsum("...np,...mq,pq->...nm", x1, x2, F)

            sum_x1 = x1.sum(dim=-1)
            sum_x2 = x2.sum(dim=-1)
            D = torch.einsum("...n,...m->...nm", sum_x1, sum_x2)
            assert K.shape == D.shape
            D = D.clamp(min=1)

            # Take average
            K /= D

            return K


class DeepEmbeddingKernel(gpytorch.kernels.Kernel):
    r"""
    https://proceedings.mlr.press/v108/buathong20a/buathong20a.pdf

    """

    is_stationary = False

    def __init__(self, ds_kernel: gpytorch.kernels.Kernel, **kwargs):
        super().__init__(**kwargs)
        self.ds_kernel = ds_kernel

        self.register_parameter(
            name="raw_inner_scale",
            parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, 1)),
        )
        self.register_constraint(
            "raw_inner_scale",
            constraint=Interval(lower_bound=0.0, upper_bound=math.sqrt(2)),
        )

        self.register_parameter(
            name="raw_outer_scale",
            parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, 1)),
        )
        self.register_constraint("raw_outer_scale", constraint=Positive())

    @property
    def inner_scale(self):
        return self.raw_inner_scale_constraint.transform(self.raw_inner_scale)

    @inner_scale.setter
    def inner_scale(self, value):
        self._set_inner_scale(value)

    def _set_inner_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_inner_scale)
        self.initialize(
            raw_inner_scale=self.raw_inner_scale_constraint.inverse_transform(value)
        )

    @property
    def outer_scale(self):
        return self.raw_outer_scale_constraint.transform(self.raw_outer_scale)

    @outer_scale.setter
    def outer_scale(self, value):
        self._set_outer_scale(value)

    def _set_outer_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outer_scale)
        self.initialize(
            raw_outer_scale=self.raw_outer_scale_constraint.inverse_transform(value)
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ):
        k11 = self.ds_kernel.forward(
            x1, x1, diag=True, last_dim_is_batch=last_dim_is_batch, **params
        )
        k11 = torch.unsqueeze(k11, -1)
        k22 = self.ds_kernel.forward(
            x2, x2, diag=True, last_dim_is_batch=last_dim_is_batch, **params
        )
        k22 = torch.unsqueeze(k22, -2)
        k12 = self.ds_kernel.forward(
            x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )
        kde = self.outer_scale * (-self.inner_scale * (k11 + k22 - 2 * k12)).exp()
        return kde


def get_model(
    features: torch.Tensor,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    state_dict=None,
) -> botorch.models.SingleTaskGP:
    r"""
    モデルを取得する。

    Parameters
    ----------
    features: torch.Tensor
        特徴量行列。
        この行列の各行が集合の要素を表す。
        この行列の各行から部分集合を選ぶ。
    train_X: torch.Tensor
        既知の部分集合。
        各行が過去に選ばれた部分集合を表す。
        0/1の行列であり、1が選ばれた要素を表す。
        train_Xの列数はfeaturesの行数と一致する。
    train_Y: torch.Tensor
        既知の部分集合の目的関数値。
    state_dict: dict or None
        モデルの状態辞書。
        以前の訓練結果を再利用する場合に指定する。

    """

    # カーネルの作成
    # DeepEmbeddingKernel と SumKernel の積のカーネルを使う。
    # SumKernel は集合の要素数の類似度を基にしたカーネルである。
    # DeepEmbeddingKernel は集合の要素数を考慮しないので SumKernel を追加している。
    ds_kernel = DoubleSumKernel(features=features)
    de_kernel = DeepEmbeddingKernel(ds_kernel=ds_kernel)
    sum_kernel = SumKernel()
    k = sum_kernel * de_kernel

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=torch.full_like(train_Y, 1e-5),
        covar_module=k,
        input_transform=None,
        outcome_transform=Standardize(m=1),
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.train()
    fit_gpytorch_mll(mll)
    model.eval()

    return model


def optimize_acqf(
    acqf: botorch.acquisition.AcquisitionFunction,
    n: int,
    min_num: int,
    max_num: int,
    width: int,
) -> tuple[list[int], float]:
    r"""
    獲得関数をビームサーチで最適化する

    Parameters
    ----------
    acqf: botorch.acquisition.AcquisitionFunction
        獲得関数。
        この関数は、0/1のリストを入力とし、スカラー値を出力する。
    n: int
        部分集合を選ぶ集合の要素数。
    min_num: int
        選ぶ部分集合の最小の要素数。
    max_num: int
        選ぶ部分集合の最大の要素数。
    width: int
        ビーム幅。

    Returns
    -------
    tuple[list[int], float]
        最適な部分集合とその獲得関数の値。

    """
    assert width >= 1
    assert min_num >= 1 and min_num <= max_num
    assert max_num <= n

    best = [0 for _ in range(n)]
    best_val = float("-inf")

    cur_cand = torch.Tensor(
        [[0 if j != i else 1 for j in range(n)] for i in range(n)]
    ).to(dtype=torch.float64)
    acqf_val = acqf.forward(cur_cand.unsqueeze(1))
    assert acqf_val.shape == torch.Size([len(cur_cand)])
    sorted_idx = acqf_val.argsort().flip(dims=[0])
    if len(sorted_idx) > width:
        sorted_idx = sorted_idx[:width]
    cur_cand = cur_cand[sorted_idx, :]
    acqf_val = acqf_val[sorted_idx]
    if min_num <= 1 and 1 <= max_num and best_val < acqf_val[0]:
        best_val = acqf_val[0].item()
        best = cur_cand[0, :].tolist()
    for i in range(2, max_num + 1):
        next_cand = []
        for c in cur_cand:
            for j in range(n):
                if c[j].item() == 0:
                    c2 = c.clone()
                    c2[j] = 1
                    next_cand.append(c2)
        next_cand = torch.stack(next_cand)
        acqf_val = acqf.forward(next_cand.unsqueeze(1))
        assert acqf_val.shape == torch.Size([len(next_cand)])
        sorted_idx = acqf_val.argsort().flip(dims=[0])
        if len(sorted_idx) > width:
            sorted_idx = sorted_idx[:width]
        next_cand = next_cand[sorted_idx, :]
        acqf_val = acqf_val[sorted_idx]
        if min_num <= i and i <= max_num and best_val < acqf_val[0]:
            best_val = acqf_val[0].item()
            best = next_cand[0, :].tolist()
        cur_cand = next_cand

    return best, best_val


def get_next_query_by_ei(
    features: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    min_num: int,
    max_num: int,
    beam_width: int,
) -> tuple[list[int], float]:
    r"""
    部分集合を選ぶベイズ最適化を行って、次のクエリを選ぶ。

    Parameters
    ----------
    features: torch.Tensor
        特徴量行列。
        この行列の各行が集合の要素を表す。
        この行列の各行から部分集合を選ぶ。
    train_x: torch.Tensor
        既知の部分集合。
        各行が過去に選ばれた部分集合を表す。
        0/1の行列であり、1が選ばれた要素を表す。
        train_xの列数はfeaturesの行数と一致する。
    train_y: torch.Tensor
        既知の部分集合の目的関数値。
    min_num: int
        選ぶ部分集合の最小の要素数。
    max_num: int
        選ぶ部分集合の最大の要素数。
    beam_width: int
        ビーム幅。

    Returns
    -------
    tuple[list[int], float]
        次のクエリの期待改善値の対数。
        次のクエリは、0/1のリストであり、1が選ばれた要素を表す。

    """
    model = get_model(features, train_x, train_y)
    nums = train_x.sum(dim=-1)
    flg = (nums >= min_num) & (nums <= max_num)
    best_f = train_y[flg].min().item()
    acqf = LogExpectedImprovement(model, best_f=best_f, maximize=False)
    next, next_ei = optimize_acqf(
        acqf=acqf, n=len(features), min_num=min_num, max_num=max_num, width=beam_width
    )
    return next, next_ei


class LogEHVI:
    r"""
    期待超体積改善値の対数を計算するクラス。

    本来は
    botorch.acquisition.multi_objective.base.MultiObjectiveAnalyticAcquisitionFunction
    を継承したクラスにするべきだが面倒なので独自に実装している。

    Parameters
    ----------
    model: SingleTaskGP
        モデル。
    n_min: int
        選ぶ部分集合の最小の要素数。
    n_max: int
        選ぶ部分集合の最大の要素数。
    best_f: list[float]
        過去の目的関数値の最小値のリスト。
        要素数はn_max - n_min + 1。

    """

    def __init__(
        self, model: SingleTaskGP, n_min: int, n_max: int, best_f: list[float]
    ):
        assert n_min <= n_max
        assert n_max - n_min + 1 == len(best_f)
        self.model = model

        self.n_min = n_min
        self.n_max = n_max
        self.best_f = best_f.copy()
        for i in range(1, len(self.best_f)):
            self.best_f[i] = min(self.best_f[i], self.best_f[i - 1])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        nums = X.sum(dim=-1)
        assert (nums >= self.n_min).all()
        assert (nums <= self.n_max).all()
        min_num = int(nums.min().item())

        new_shape = X.shape[:-1] + (self.n_max - min_num + 1,)
        acq_vals = torch.full(new_shape, fill_value=float("-inf")).to(
            dtype=torch.float64
        )
        for tmp_num in range(min_num, self.n_max + 1):
            tmp_i = tmp_num - self.n_min
            tmp_j = tmp_num - min_num
            tmp_best_f = self.best_f[tmp_i]
            logei = LogExpectedImprovement(
                model=self.model,
                best_f=tmp_best_f,
                maximize=False,
            )
            mask = nums <= tmp_num
            acq_vals[..., tmp_j][mask] = logei.forward(X[mask].unsqueeze(1)).squeeze(-1)
        acq_vals = acq_vals.logsumexp(dim=-1).squeeze(-1)
        return acq_vals


def get_next_query_by_mo(
    features: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    min_num: int,
    max_num: int,
    beam_width: int,
) -> tuple[list[int], float]:
    r"""
    部分集合を選ぶベイズ最適化を行って、次のクエリを選ぶ。

    get_next_query_by_ei とは異なり、集合の要素数と目的関数値との多目的最適化を行う。
    集合の要素数が大きくなるほど目的関数値は小さくなると仮定している。

    Parameters
    ----------
    features: torch.Tensor
        特徴量行列。
        この行列の各行が集合の要素を表す。
        この行列の各行から部分集合を選ぶ。
    train_x: torch.Tensor
        既知の部分集合。
        各行が過去に選ばれた部分集合を表す。
        0/1の行列であり、1が選ばれた要素を表す。
        train_xの列数はfeaturesの行数と一致する。
    train_y: torch.Tensor
        既知の部分集合の目的関数値。
    min_num: int
        選ぶ部分集合の最小の要素数。
    max_num: int
        選ぶ部分集合の最大の要素数。
    beam_width: int
        ビーム幅。

    Returns
    -------
    tuple[list[int], float]
        次のクエリの期待超体積改善値の対数。
        次のクエリは、0/1のリストであり、1が選ばれた要素を表す。

    """
    assert beam_width >= 1
    assert min_num >= 1 and min_num <= max_num
    assert max_num <= len(features)

    model = get_model(features, train_x, train_y)

    logei = LogExpectedImprovement(model, best_f=train_y.min().item(), maximize=False)

    nums = train_x.sum(dim=-1)
    best_fs = [float("inf") for _ in range(max_num - min_num + 1)]
    unums = nums.unique().to(dtype=torch.int32).tolist()
    for i, tmp_n in enumerate(unums):
        best_fs[tmp_n - min_num] = min(
            best_fs[tmp_n - min_num], train_y[nums == tmp_n, :].min().item()
        )

    lehvi = LogEHVI(model, min_num, max_num, best_fs)

    best = [0 for _ in range(len(features))]
    best_val = float("-inf")

    cur_cand = torch.Tensor(
        [
            [0 if j != i else 1 for j in range(len(features))]
            for i in range(len(features))
        ]
    ).to(dtype=torch.float64)

    if min_num <= 1 and 1 <= max_num:
        acqf_val = lehvi.forward(cur_cand.unsqueeze(1))
    else:
        acqf_val = logei.forward(cur_cand.unsqueeze(1))
    assert acqf_val.shape == torch.Size([len(cur_cand)])

    sorted_idx = acqf_val.argsort().flip(dims=[0])
    if len(sorted_idx) > beam_width:
        sorted_idx = sorted_idx[:beam_width]

    cur_cand = cur_cand[sorted_idx, :]
    acqf_val = acqf_val[sorted_idx]

    if min_num <= 1 and 1 <= max_num and best_val < acqf_val[0]:
        best_val = acqf_val[0].item()
        best = cur_cand[0, :].tolist()

    for i in range(2, max_num + 1):
        next_cand = []
        for c in cur_cand:
            for j in range(len(features)):
                if c[j].item() == 0:
                    c2 = c.clone()
                    c2[j] = 1
                    next_cand.append(c2)
        next_cand = torch.stack(next_cand)
        if min_num <= i and i <= max_num:
            acqf_val = lehvi.forward(next_cand.unsqueeze(1))
        else:
            acqf_val = logei.forward(next_cand.unsqueeze(1))
        assert acqf_val.shape == torch.Size([len(next_cand)])
        sorted_idx = acqf_val.argsort().flip(dims=[0])
        if len(sorted_idx) > beam_width:
            sorted_idx = sorted_idx[:beam_width]
        next_cand = next_cand[sorted_idx, :]
        acqf_val = acqf_val[sorted_idx]
        if min_num <= i and i <= max_num and best_val < acqf_val[0]:
            best_val = acqf_val[0].item()
            best = next_cand[0, :].tolist()
        cur_cand = next_cand

    return best, best_val
