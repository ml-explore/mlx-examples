import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils
import math

class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Calculate the step size for the grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        # Create a grid of points for the splines
        grid = (
            (mx.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).reshape(-1, 1)  # Reshape to a column vector
        )
        # Tile the grid for each input feature
        self.grid = mx.tile(grid, (1, in_features))

        # Initialize weights for the base layer and the spline layer
        self.base_weight = mx.random.uniform(shape=(out_features, in_features))
        self.spline_weight = mx.random.uniform(shape=(out_features, in_features, grid_size + spline_order))

        # Initialize scaler for spline weights if standalone scaling is enabled
        if enable_standalone_scale_spline:
            self.spline_scaler = mx.random.uniform(shape=(out_features, in_features))

        # Save the scaling factors and activation function
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()