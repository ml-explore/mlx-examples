import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math


def B_batch(x, grid, k=0, extend=True):
    """
    evaludate x on B-spline bases

    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True

    Returns:
    --------
        spline values : 3D torch.tensor
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.
    """

    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = mx.concatenate([grid[:, [0]] - h, grid], axis=1)
            grid = mx.concatenate([grid, grid[:, [-1]] + h], axis=1)
        return grid

    if extend:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2)
    x = x.unsqueeze(dim=1)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(
            x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False
        )
        value = (x - grid[:, : -(k + 1)]) / (
            grid[:, k:-1] - grid[:, : -(k + 1)]
        ) * B_km1[:, :-1] + (grid[:, k + 1 :] - x) / (
            grid[:, k + 1 :] - grid[:, 1:(-k)]
        ) * B_km1[
            :, 1:
        ]
    return value


def curve2coef(x_eval, y_eval, grid, k):
    """
    converting B-spline curves to B-spline coefficients using least squares.

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        grid : 2D torch.tensor
            shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
    """
    # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
    mat = B_batch(x_eval, grid, k).permute(0, 2, 1)

    coef = np.linalg.lstsq(
        mat,
        y_eval.unsqueeze(dim=2)
    ).solution[:, :, 0]
    return coef


def coef2curve(x_eval, grid, coef, k):
    """
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).

    Args:
    -----
        x_eval : 2D torch.tensor)
            shape (number of splines, number of samples)
        grid : 2D torch.tensor)
            shape (number of splines, number of grid points)
        coef : 2D torch.tensor)
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.

    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
    """
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    y_eval = np.einsum("ij,ijk->ik", coef, B_batch(x_eval, grid, k))
    return y_eval


class KANLayer(nn.Module):
    def __init__(
        self,
        in_dim=3,
        out_dim=2,

        num=5,
        k=3,

        noise_scale=0.1,
        scale_base=1.0,
        scale_sp=1.0,

        base_fun=nn.SiLU,

        grid_eps=0.02,
        grid_range=[-1, 1],

        sp_trainable=True,
        sb_trainable=True,
    ):
        """'
        initialize a KANLayer

        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base : float
                the scale of the residual function b(x). Default: 1.0.
            scale_sp : float
                the scale of the base function spline(x). Default: 1.0.
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
        """
        super(KANLayer, self).__init__()
        self.size = size = out_dim * in_dim

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        # shape: (size, num)
        self.grid = np.einsum(
            "i,j->ij",
            np.ones(size),
            np.linspace(start=grid_range[0], stop=grid_range[1], num=num + 1),
        )

        self.noises = (
            (np.random.rand(size, self.grid.shape[1]) - 1 / 2) * noise_scale / num
        )

        # shape: (size, coef)
        self.coef = curve2coef(self.grid, self.noises, self.grid, k)

        if isinstance(scale_base, float):
            self.scale_base = mx.ones(size) * scale_base
        else:
            self.scale_base = scale_base

        self.scale_sp = mx.ones(size) * scale_sp
        self.base_fun = base_fun
        self.mask = mx.ones(size)
        self.grid_eps = grid_eps
        self.weight_sharing = np.arange(size)
        self.lock_counter = 0
        self.lock_id = mx.zeros(size)


    def __call__(self, x):
        """
        KANLayer __call__ given input x

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs
        """
        batch = x.shape[0]
        # x: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        x = (
            np.einsum(
                "ij,k->ikj",
                x,
                np.ones(self.out_dim)
            )
            .reshape(batch, self.size)
            .permute(1, 0)
        )

        preacts = (
            x.permute(1, 0).clone().reshape(batch, self.out_dim, self.in_dim)
        )

        base = self.base_fun(x).moveaxis(1, 0)  # shape (batch, size) --- .moveaxis(1, 0)

        y = coef2curve(
            x_eval=x,
            grid=self.grid[self.weight_sharing],
            coef=self.coef[self.weight_sharing],
            k=self.k,
        )  # shape (size, batch)

        y = y.permute(1, 0)  # shape (batch, size)
        postspline = y.clone().reshape(batch, self.out_dim, self.in_dim)

        y = (
            self.scale_base.unsqueeze(dim=0) * base
            + self.scale_sp.unsqueeze(dim=0) * y
        )

        y = self.mask[None, :] * y
        postacts = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = mx.sum(y.reshape(batch, self.out_dim, self.in_dim), axis=2)  # shape (batch, out_dim)

        # y shape: (batch, out_dim); preacts shape: (batch, in_dim, out_dim)
        # postspline shape: (batch, in_dim, out_dim); postacts: (batch, in_dim, out_dim)
        # postspline is for extension; postacts is for visualization
        return y, preacts, postacts, postspline


    def update_grid_from_samples(self, x):
        """
        update grid from samples

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
        """
        batch = x.shape[0]
        x = (
            np.einsum(
                "ij,k->ikj",
                x,
                np.ones(self.out_dim,),
            )
            .reshape(batch, self.size)
            .permute(1, 0)
        )
        x_pos = mx.sort(x, axis=1)[0]

        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)

        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]

        grid_adaptive = x_pos[:, ids]
        margin = 0.01

        grid_uniform = mx.concatenate(
            [
                grid_adaptive[:, [0]]
                - margin
                + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin)
                * a
                for a in np.linspace(0, 1, num=self.grid.shape[1])
            ],
            axis=1,
        )

        self.grid.data = (
            self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        )

        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)


    def initialize_grid_from_parent(self, parent, x):
        """
        update grid from a parent KANLayer & samples

        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
        """
        batch = x.shape[0]
        # preacts: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        x_eval = (
            np.einsum(
                "ij,k->ikj",
                x,
                np.ones(self.out_dim)
            )
            .reshape(batch, self.size)
            .permute(1, 0)
        )

        x_pos = parent.grid

        sp2 = KANLayer(
            in_dim=1,
            out_dim=self.size,
            k=1,
            num=x_pos.shape[1] - 1,
            scale_base=0.0,
        )

        sp2.coef.data = curve2coef(sp2.grid, x_pos, sp2.grid, k=1)

        y_eval = coef2curve(x_eval, parent.grid, parent.coef, parent.k)

        percentile = mx.linspace(-1, 1, self.num + 1)

        self.grid.data = sp2(percentile.unsqueeze(dim=1))[0].permute(1, 0)
        self.coef.data = curve2coef(x_eval, y_eval, self.grid, self.k)


    def get_subset(self, in_id, out_id):
        """
        get a smaller KANLayer from a larger KANLayer (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns:
        --------
            spb : KANLayer
        """
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)

        spb.grid.data = self.grid.reshape(
            self.out_dim, self.in_dim, spb.num + 1
        )[out_id][:, in_id].reshape(-1, spb.num + 1)

        spb.coef.data = self.coef.reshape(
            self.out_dim, self.in_dim, spb.coef.shape[1]
        )[out_id][:, in_id].reshape(-1, spb.coef.shape[1])

        spb.scale_base.data = self.scale_base.reshape(
            self.out_dim, self.in_dim
        )[out_id][:, in_id].reshape(
            -1,
        )

        spb.scale_sp.data = self.scale_sp.reshape(self.out_dim, self.in_dim)[
            out_id
        ][:, in_id].reshape(
            -1,
        )

        spb.mask.data = self.mask.reshape(self.out_dim, self.in_dim)[out_id][
            :, in_id
        ].reshape(
            -1,
        )

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        spb.size = spb.in_dim * spb.out_dim
        return spb


    def lock(self, ids):
        """
        lock activation functions to share parameters based on ids

        Args:
        -----
            ids : list
                list of ids of activation functions

        """
        self.lock_counter += 1
        # ids: [[i1,j1],[i2,j2],[i3,j3],...]
        for i in range(len(ids)):
            if i != 0:
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = (
                    ids[0][1] * self.in_dim + ids[0][0]
                )
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = (
                self.lock_counter
            )


    def unlock(self, ids):
        """
        unlock activation functions

        Args:
        -----
            ids : list
                list of ids of activation functions
        """
        # check ids are locked
        num = len(ids)
        locked = True
        for i in range(num):
            locked *= (
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]]
                == self.weight_sharing[ids[0][1] * self.in_dim + ids[0][0]]
            )
        if not locked:
            print("they are not locked. unlock failed.")
            return 0
        for i in range(len(ids)):
            self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = (
                ids[i][1] * self.in_dim + ids[i][0]
            )
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = 0
        self.lock_counter -= 1


class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        num=5,
        k=3,
        noise_scale=0.1,
        scale_base=1.0,
        scale_sp=1.0,
        base_fun=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.num = num
        self.k = k

        self.layers = []

        for in_dim, out_dim in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLayer(
                    in_dim,
                    out_dim,
                    num=num,
                    k=k,
                    noise_scale=noise_scale,
                    scale_base=scale_base,
                    scale_sp=scale_sp,
                    base_fun=base_fun,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def __call__(self, x: mx.array, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
