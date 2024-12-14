import mnist

import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Generator Block
def GenBlock(in_dim:int,out_dim:int):
    return nn.Sequential(
        nn.Linear(in_dim,out_dim),
        nn.BatchNorm(out_dim, 0.8),
        nn.LeakyReLU(0.2)
    )

# Generator Model
class Generator(nn.Module):

    def __init__(self, z_dim:int = 32, im_dim:int = 784, hidden_dim: int = 256):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            GenBlock(z_dim, hidden_dim),
            GenBlock(hidden_dim, hidden_dim * 2),
            GenBlock(hidden_dim * 2, hidden_dim * 4),

            nn.Linear(hidden_dim * 4,im_dim),
        )
        
    def __call__(self, noise):
        x = self.gen(noise)
        return mx.tanh(x)

# make 2D noise with shape n_samples x z_dim
def get_noise(n_samples:list[int], z_dim:int)->list[int]:
    return mx.random.normal(shape=(n_samples, z_dim))

#---------------------------------------------#

# Discriminator Block
def DisBlock(in_dim:int,out_dim:int):
    return nn.Sequential(
        nn.Linear(in_dim,out_dim),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Dropout(0.3),
    )

# Discriminator Model
class Discriminator(nn.Module):

    def __init__(self,im_dim:int = 784, hidden_dim:int = 256):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            DisBlock(im_dim, hidden_dim * 4),
            DisBlock(hidden_dim * 4, hidden_dim * 2),
            DisBlock(hidden_dim * 2, hidden_dim),
            
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )
        
    def __call__(self, noise):
        return self.disc(noise)
    
# Discriminator Loss
def disc_loss(gen, disc, real, num_images, z_dim):
    
    noise =  mx.array(get_noise(num_images, z_dim))
    fake_images = gen(noise)
        
    fake_disc = disc(fake_images)
    
    fake_labels = mx.zeros((fake_images.shape[0],1))
            
    fake_loss = mx.mean(nn.losses.binary_cross_entropy(fake_disc,fake_labels,with_logits=True))
        
    real_disc = mx.array(disc(real))
    real_labels = mx.ones((real.shape[0],1))
    
    real_loss = mx.mean(nn.losses.binary_cross_entropy(real_disc,real_labels,with_logits=True))
    
    disc_loss = (fake_loss + real_loss) / 2.0

    return disc_loss

# Genearator Loss
def gen_loss(gen, disc, num_images, z_dim):

    noise = mx.array(get_noise(num_images, z_dim))
    
    fake_images = gen(noise)
    fake_disc = mx.array(disc(fake_images))

    fake_labels = mx.ones((fake_images.shape[0],1))
            
    gen_loss = nn.losses.binary_cross_entropy(fake_disc,fake_labels,with_logits=True)
    
    return mx.mean(gen_loss)

# make batch of images
def batch_iterate(batch_size: int, ipt: list[int])-> list[int]:
    perm = np.random.permutation(len(ipt))
    for s in range(0, len(ipt), batch_size):
        ids = perm[s : s + batch_size]
        yield ipt[ids]

# plot batch of images at epoch steps
def show_images(epoch_num:int,imgs:list[int],num_imgs:int = 25):
    if (imgs.shape[0] > 0): 
        fig,axes = plt.subplots(5, 5, figsize=(5, 5))
        
        for i, ax in enumerate(axes.flat):
            img = mx.array(imgs[i]).reshape(28,28)
            ax.imshow(img,cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('gen_images/img_{}.png'.format(epoch_num))
        plt.show()

def main(args:dict):
    seed = 42
    n_epochs = 500
    z_dim = 128
    batch_size = 128
    lr = 2e-5
    
    mx.random.seed(seed)

    # Load the data
    train_images,*_ = map(np.array, getattr(mnist,'mnist')())
    
    # Normalization images => [-1,1]
    train_images = train_images * 2.0 - 1.0
    
    gen = Generator(z_dim)
    mx.eval(gen.parameters())
    gen_opt = optim.Adam(learning_rate=lr, betas=[0.5, 0.999])

    disc = Discriminator()
    mx.eval(disc.parameters())
    disc_opt = optim.Adam(learning_rate=lr, betas=[0.5, 0.999])

    # TODO training...

    D_loss_grad = nn.value_and_grad(disc, disc_loss)
    G_loss_grad = nn.value_and_grad(gen, gen_loss)

    for epoch in tqdm(range(n_epochs)):

        for idx,real in enumerate(batch_iterate(batch_size, train_images)):
                    
            # TODO Train Discriminator
            D_loss,D_grads = D_loss_grad(gen, disc,mx.array(real), batch_size, z_dim)

            # Update optimizer
            disc_opt.update(disc, D_grads)
            
            # Update gradients
            mx.eval(disc.parameters(), disc_opt.state)

            # TODO Train Generator
            G_loss,G_grads = G_loss_grad(gen, disc, batch_size, z_dim)
            
            # Update optimizer
            gen_opt.update(gen, G_grads)
            
            # Update gradients
            mx.eval(gen.parameters(), gen_opt.state)        
            
        if epoch%100==0:
                print("Epoch: {}, iteration: {}, Discriminator Loss:{}, Generator Loss: {}".format(epoch,idx,D_loss,G_loss))
                fake_noise = mx.array(get_noise(batch_size, z_dim))
                fake = gen(fake_noise)
                show_images(epoch,fake)

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
