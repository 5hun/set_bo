import numpy as np
import scipy as sp
import torch
import pulp
from sklearn.metrics.pairwise import euclidean_distances
import pulp


def obj(features: torch.Tensor, x: list[int]):
    r"""
    Calculate the sum of the minimum distance between the selected points and all points.
    """
    assert len(x) == len(features)
    x = np.array(x)
    idx = np.nonzero(x)[0]
    centers = features[idx, :]
    dists = sp.spatial.distance_matrix(centers, features)
    return float(dists.min(axis=0).sum())


def solve_by_mip(features: torch.Tensor, num_depot: int) -> tuple[float, list[int]]:
    f_distmat = euclidean_distances(features)

    model = pulp.LpProblem("facility_location", pulp.LpMinimize)
    x = [
        pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        for i in range(len(features))
    ]
    y = [
        [
            pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1, cat=pulp.LpBinary)
            for j in range(len(features))
        ]
        for i in range(len(features))
    ]
    model += pulp.lpSum([x[i] for i in range(len(features))]) <= num_depot
    for i in range(len(features)):
        model += pulp.lpSum([y[i][j] for j in range(len(features))]) == 1
    for i in range(len(features)):
        for j in range(len(features)):
            model += y[i][j] <= x[j]
    model += pulp.lpSum(
        f_distmat[i, j] * y[i][j]
        for i in range(len(features))
        for j in range(len(features))
    )
    status = model.solve()
    assert status == pulp.LpStatusOptimal
    return (
        model.objective.value(),
        [i for i in range(len(features)) if x[i].value() == 1],
    )
