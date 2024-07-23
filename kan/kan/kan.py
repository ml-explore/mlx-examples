import mlx.core as mx
import mlx.nn as nn

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
        hidden_act=nn.SiLU,
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
        self.hidden_act = hidden_act()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize base_weight with random values scaled by scale_base
        self.base_weight = mx.random.uniform(shape=self.base_weight.shape) * self.scale_base

        # Initialize spline_weight with random values scaled by scale_spline
        self.spline_weight = mx.random.uniform(shape=self.spline_weight.shape) * self.scale_spline

        if self.enable_standalone_scale_spline:
            # Initialize spline_scaler with random values scaled by scale_spline
            self.spline_scaler = mx.random.uniform(shape=self.spline_scaler.shape) * self.scale_spline


    def b_splines(self, x):
        # Ensure input x has the correct dimensions
        assert x.ndim == 2 and x.shape[1] == self.in_features

        x = x[:, :, None]  # Add a new dimension to x
        grid_reshaped = self.grid[:-1].T[None, :, :]  # Reshape to (1, in_features, grid_size)
        next_grid_reshaped = self.grid[1:].T[None, :, :]  # Reshape to (1, in_features, grid_size)

        # Compute the base splines
        bases = mx.logical_and(x >= grid_reshaped, x < next_grid_reshaped).astype(x.dtype)
        for k in range(1, self.spline_order + 1):
            # Update the splines for each order
            bases = (
                (x - self.grid[:-(k+1)].T[None, :, :]) /
                (self.grid[k:-1].T[None, :, :] - self.grid[:-(k+1)].T[None, :, :]) *
                bases[:, :, :-1]
            ) + (
                (self.grid[(k+1):].T[None, :, :] - x) /
                (self.grid[(k+1):].T[None, :, :] - self.grid[1:(-k)].T[None, :, :]) *
                bases[:, :, 1:]
            )
        # Ensure the bases have the correct shape
        assert bases.shape == (x.shape[0], self.in_features, self.grid_size + self.spline_order)
        return bases

        
    def curve2coeff(self, x, y):
        # Ensure input x and y have the correct dimensions
        assert x.ndim == 2 and x.shape[1] == self.in_features
        assert y.shape == (x.shape[0], self.in_features, self.out_features)
        
        # Transpose and reshape the splines and y for solving the least squares problem
        A = self.b_splines(x).transpose(1, 0, 2)
        B = y.transpose(1, 0, 2)  
        solution = mx.linalg.lstsq(A, B).solution
        result = solution.transpose(0, 2, 1)

        # Ensure the result has the correct shape
        assert result.shape == (
            self.out_features,
            self.in_features,  
            self.grid_size + self.spline_order,
        )
        return result


# loschen und commit
    @property
    def scaled_spline_weight(self):
        # Scale the spline weight if standalone scaling is enabled
        return self.spline_weight * (
            self.spline_scaler[:, :, None] 
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def __call__(self, x):
        # Ensure input x has the correct dimensions
        assert x.ndim == 2 and x.shape[1] == self.in_features

        # Compute the base output
        base_output = mx.matmul(self.hidden_act(x), self.base_weight.T)

        # Compute the spline output
        spline_output = mx.matmul(
            self.b_splines(x.reshape(x.shape[0], -1)).reshape(x.shape[0], -1),
            self.scaled_spline_weight.reshape(self.out_features, -1).T,  
        )

        # Return the sum of the base output and the spline output
        return base_output + spline_output

    def update_grid(self, x, margin=0.01):
        # Ensure input x has the correct dimensions
        assert x.ndim == 2 and x.shape[1] == self.in_features
        batch = x.shape[0]
        
        # Compute splines and their outputs
        splines = self.b_splines(x)
        splines = splines.transpose(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.transpose(1, 2, 0) 
        unreduced_spline_output = mx.matmul(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.transpose(1, 0, 2)
        
        # Sort x and create adaptive and uniform grids
        x_sorted = mx.sort(x, axis=0)
        grid_adaptive = x_sorted[
            mx.linspace(
                0, batch - 1, self.grid_size + 1, dtype=mx.int32
            ).astype(mx.int64) 
        ]
        
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            mx.arange(self.grid_size + 1).reshape(-1, 1).astype(mx.float32)
            * uniform_step
            + x_sorted[0] 
            - margin
        )
        
        # Blend the adaptive and uniform grids
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = mx.concatenate(
            [
                grid[:1] 
                - uniform_step
                * mx.arange(self.spline_order, 0, -1).reshape(-1, 1),
                grid,
                grid[-1:]
                + uniform_step 
                * mx.arange(1, self.spline_order + 1).reshape(-1, 1),
            ],
            axis=0,
        )
        
        # Update the grid and spline weights
        self.grid.set_value(grid.T)
        self.spline_weight.set_value(self.curve2coeff(x, unreduced_spline_output))
        
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # Compute L1 norm of the spline weights
        l1_fake = mx.abs(self.spline_weight).mean(axis=-1)
        regularization_loss_activation = l1_fake.sum()

        # Compute entropy of the spline weights
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -mx.sum(p * mx.log(p))

        # Return the combined regularization loss
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

        
class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,  
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        hidden_act=nn.SiLU,
        grid_eps=0.02, 
        grid_range=[-1, 1],
    ):
        super().__init__()

        # Save the grid and spline parameters
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        self.layers = [] # Initialize the list of layers

        # Create KANLinear layers based on the hidden layers provided
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    hidden_act=hidden_act,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
    
    def __call__(self, x, update_grid=False):
        # Pass the input x through each layer
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x) # Update the grid if required
            x = layer(x) # Pass the input through the current layer
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # Compute the regularization loss for all layers
        return mx.add(*(
            layer.regularization_loss(regularize_activation, regularize_entropy) 
            for layer in self.layers
        ))