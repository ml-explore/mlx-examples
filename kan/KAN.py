import mlx.core as mx
import mlx.nn as nn

class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, base_activation=nn.GELU, grid_range=[-1, 1]):
        super(KAN, self).__init__()
        # List of hidden layer dimensions for the neural network.
        self.layers_hidden = layers_hidden
        # The number of points in the grid for the spline interpolation.
        self.grid_size = grid_size
        # The order of the spline used in the interpolation.
        self.spline_order = spline_order
        # Activation function used for the initial transformation of the input.
        self.base_activation = base_activation()
        # The range of values over which the grid for spline interpolation is defined.
        self.grid_range = grid_range

        # Parameters and layer norms initialization
        self.base_weights = nn.P()  # Parameters for the linear transformations in each layer.
        self.spline_weights = nn.ParameterList()  # Parameters for the spline-based transformations in each layer.
        self.layer_norms = nn.ModuleList()  # Layer normalization for each layer to ensure stable training.
        self.prelus = nn.ModuleList()  # PReLU activations for each layer to introduce non-linearity.
        self.grids = []  # Stores the computed grid values for spline calculations for each layer.

        # Loop through the layers to initialize weights, norms, and grids
        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            # Initialize the base weights with random values for the linear transformation.
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            # Initialize the spline weights with random values for the spline transformation.
            self.spline_weights.append(nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order)))
            # Add a layer normalization for stabilizing the output of this layer.
            self.layer_norms.append(nn.LayerNorm(out_features))
            # Add a PReLU activation for this layer to provide a learnable non-linearity.
            self.prelus.append(nn.PReLU())

            # Compute the grid values based on the specified range and grid size.
            h = (self.grid_range[1] - self.grid_range[0]) / grid_size
            grid = mx.linspace(
                self.grid_range[0] - h * spline_order,
                self.grid_range[1] + h * spline_order,
                grid_size + 2 * spline_order + 1,
                dtype=torch.float32
            ).expand(in_features, -1).contiguous()
            self.grids.append(grid)

        # Initialize the weights using Kaiming uniform distribution for better initial values.
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.spline_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    def forward(self, x):
        # Process each layer using the defined base weights, spline weights, norms, and activations.
        for i, (base_weight, spline_weight, layer_norm, prelu) in enumerate(zip(self.base_weights, self.spline_weights, self.layer_norms, self.prelus)):
            grid = self._buffers[f'grid_{i}']
            # Move the input tensor to the device where the weights are located.
            x = x.to(base_weight.device)

            # Perform the base linear transformation followed by the activation function.
            base_output = F.linear(self.base_activation(x), base_weight)
            x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
            # Compute the basis for the spline using intervals and input values.
            bases = ((x_uns >= grid[:, :-1]) & (x_uns < grid[:, 1:])).to(x.dtype)

            # Compute the spline basis over multiple orders.
            for k in range(1, self.spline_order + 1):
                left_intervals = grid[:, :-(k + 1)]
                right_intervals = grid[:, k:-1]
                delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals), right_intervals - left_intervals)
                bases = ((x_uns - left_intervals) / delta * bases[:, :, :-1]) + \
                        ((grid[:, k + 1:] - x_uns) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
            bases = bases.contiguous()

            # Compute the spline transformation and combine it with the base transformation.
            spline_output = F.linear(bases.view(x.size(0), -1), spline_weight.view(spline_weight.size(0), -1))
            # Apply layer normalization and PReLU activation to the combined output.
            x = prelu(layer_norm(base_output + spline_output))

        return x
