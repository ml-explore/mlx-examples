import mnist
from tqdm import tqdm

import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Generator Block
def GenBlock(in_dim:int,out_dim:int):
    return nn.Sequential(
        nn.Linear(in_dim,out_dim),
        nn.BatchNorm(out_dim),
        nn.ReLU()
    )

# Generator Layer
class Generator(nn.Module):

    def __init__(self, z_dim:int = 10, im_dim:int = 784, hidden_dim: int =128):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            GenBlock(z_dim, hidden_dim),
            GenBlock(hidden_dim, hidden_dim * 2),
            GenBlock(hidden_dim * 2, hidden_dim * 4),
            GenBlock(hidden_dim * 4, hidden_dim * 8),


            nn.Linear(hidden_dim * 8,im_dim),
            nn.Sigmoid()
        )
        
    def forward(self, noise):

        return self.gen(noise) 


# return random n,m normal distribution
def get_noise(n_samples:int, z_dim:int)->list:
    return np.random.randn(n_samples,z_dim)

#---------------------------------------------#

# Discriminator Block
def DisBlock(in_dim:int,out_dim:int):   
    return nn.Sequential(
        nn.Linear(in_dim,out_dim),
        nn.LeakyReLU(negative_slope=0.2)
    )

# Discriminator Layer
class Discriminator(nn.Module):

    def __init__(self,im_dim:int = 784, hidden_dim:int = 128):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            DisBlock(im_dim, hidden_dim * 4),
            DisBlock(hidden_dim * 4, hidden_dim * 2),
            DisBlock(hidden_dim * 2, hidden_dim),

            nn.Linear(hidden_dim,1),
        )
        
    def forward(self, noise):

        return self.disc(noise)

def main(args:dict):
    seed = 42
    criterion = nn.losses.binary_cross_entropy
    n_epochs = 200
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.00001
    
    np.random.seed(seed)

    # Load the data
    train_images, train_labels, test_images, test_labels = map(
        mx.array, getattr(mnist, args.dataset)()
    )
    
    gen = Generator(z_dim)
    gen_opt = optim.Adam(learning_rate=lr)
    disc = Discriminator()
    disc_opt = optim.Adam(learning_rate=lr)
    
    # use partial function
    def disc_loss(gen, disc, criterion, real, num_images, z_dim):
        noise = get_noise(num_images, z_dim,device)
        fake_images = gen(noise)
        
        fake_disc = disc(fake_images.detach())
        fake_labels = mx.zeros(fake_images.size(0),1)
        fake_loss = criterion(fake_disc,fake_labels)
        
        real_disc = disc(real)
        real_labels = mx.ones(real.size(0),1)
        real_loss = criterion(real_disc,real_labels)

        disc_loss = (fake_loss + real_loss) / 2

        return disc_loss
    
    def gen_loss(gen, disc, criterion, num_images, z_dim):

        noise = get_noise(num_images, z_dim,device)
        fake_images = gen(noise)
        
        fake_disc = disc(fake_images)
        fake_labels = mx.ones(fake_images.size(0),1)
        
        gen_loss = criterion(fake_disc,fake_labels)

        return gen_loss

    # TODO training...
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple GAN on MNIST with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="The dataset to use.",
    )
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    main(args)
