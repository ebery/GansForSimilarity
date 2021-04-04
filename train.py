import argparse
from torchvision.utils import save_image

from utils import *

parser = argparse.ArgumentParser(description='GAN training on FashionMNIST dataset.')
parser.add_argument('--data_path', default='data/mnist/', type=str, help='Directory of the dataset')
parser.add_argument('--results_path', default='results/', type=str, help='Results directory')
parser.add_argument('--pretrained_models_path', default='pretrained_models',
                    type=str, help='Directory for trained models')
parser.add_argument('--iterations', default=20000, type=int, help='Directory for trained models')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--dcgan_lr', default=1e-3, type=float, help='learning rate for ADAM')
parser.add_argument('--wgan_lr', default=5e-5, type=float, help='learning rate for SGD')
parser.add_argument('--latent_dim', default=100, type=int, help='dimensions of latent vector')
parser.add_argument('--clip_value', default=1e-2, type=float, help='weights clipping value')
parser.add_argument('--b1', default=0.5, type=float, help='betta1 for ADAM optimizer')
parser.add_argument('--b2', default=0.999, type=float, help='betta2 for ADAM optimizer')
parser.add_argument('--gan_type', default='dcgan', type=str, choices=['wgan', 'dcgan'], help="dcgan or wgan")
parser.add_argument('--output_dim', default=100, type=int, help='dimensions of output feature vector')
args = parser.parse_args()

GENERATED_IMAGES_PATH = os.path.join(args.results_path, "Generated Images/{0}/")
MODELS_PATH = os.path.join(args.pretrained_models_path, "{0}/")
IMAGES_PATH = os.path.join(args.results_path, "images/{0}/")
DIM = 64  # Model dimensionality
IMG_SIZE = 32
CHANNELS = 1
N_CRITIC = 5
SAMPLE_INTERVAL = 400

img_shape = (CHANNELS, IMG_SIZE, IMG_SIZE)
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def train(generator, critic, criterion, optimizer_G, optimizer_D, dataloader, mode):
    assert(mode in ['wgan', 'dcgan'])
    print(f"\nStarted {mode} training:")
    G_losses = []
    D_losses = []
    images_path = IMAGES_PATH.format(mode)
    n_critic = N_CRITIC if mode == 'wgan' else 1

    for i in range(1, args.iterations+1):
        real_batch = next(iter(dataloader))

        loss_D = z = None
        for i_critic in range(n_critic):
            # ---------------------
            #  Train Critic
            # ---------------------

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (real_batch[0].shape[0], args.latent_dim))))

            optimizer_D.zero_grad()

            # Generate a batch of images
            real_imgs = Variable(real_batch[0].type(Tensor))
            fake_imgs = generator(z).detach()
            loss_D = criterion(fake_imgs, critic, device, real_imgs)
            loss_D.backward()
            optimizer_D.step()

            if mode == 'wgan':
                # Clip weights of critic
                for p in critic.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z)
        loss_G = criterion(gen_imgs, critic, device)
        loss_G.backward()
        optimizer_G.step()
        if i % 100 == 0:
            print(
                "[Iteration %d/%d] [D loss: %f] [G loss: %f]"
                % (i, args.iterations, loss_D.item(), loss_G.item())
            )

        G_losses.append(loss_G)
        D_losses.append(loss_D)

        if i % SAMPLE_INTERVAL == 0:
            save_image(gen_imgs.data, f"{images_path}{i}.png", nrow=8, normalize=True)

    G_losses = np.asarray(G_losses)
    D_losses = np.asarray(D_losses)
    np.savetxt(os.path.join(args.results_path, f"{mode}_losses_g.csv"), G_losses, delimiter=",")
    np.savetxt(os.path.join(args.results_path, f"{mode}_losses_d.csv"), D_losses, delimiter=",")
    torch.save(generator.state_dict(), f'{MODELS_PATH.format(mode)}{mode}_g.pkl')
    torch.save(critic.state_dict(), f'{MODELS_PATH.format(mode)}{mode}_d.pkl')


def train_and_save_model(dataloader, mode):
    # Initialize generator and critic
    generator, critic = get_models(mode, latent_dim=args.latent_dim, model_dim=DIM, device=device,
                                   output_dim=args.output_dim, init=True)
    criterion = adversarial_loss if mode == 'dcgan' else wasserstein_loss
    # Optimizers
    optimizer_G, optimizer_D = get_optimizers(generator, critic, mode, args.dcgan_lr, args.wgan_lr, (args.b1, args.b2))
    dirs_path_list = [MODELS_PATH.format(mode), GENERATED_IMAGES_PATH.format(mode), IMAGES_PATH.format(mode)]
    create_dirs(dirs_path_list)
    train(generator, critic, criterion, optimizer_G, optimizer_D, dataloader, mode)


if __name__ == '__main__':
    train_and_save_model(get_dataloader(args.data_path, IMG_SIZE, args.batch_size), args.gan_type)
    plot_losses(args.results_path, args.gan_type)
    plt.show()
