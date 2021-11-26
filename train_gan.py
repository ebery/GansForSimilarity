import argparse
from torchvision.utils import save_image

from utils import *

parser = argparse.ArgumentParser(description='GAN training on Cifar10 dataset.')
parser.add_argument('--data_path', default='data/cifar10/', type=str, help='Directory of the dataset')
parser.add_argument('--results_path', default='results/', type=str, help='Results directory')
parser.add_argument('--pretrained_models_path', default='pretrained_models',
                    type=str, help='Directory for trained models')
parser.add_argument('--epochs', default=200, type=int, help='Directory for trained models')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate for ADAM')
parser.add_argument('--latent_dim', default=100, type=int, help='dimensions of latent vector')
parser.add_argument('--b1', default=0.5, type=float, help='betta1 for ADAM optimizer')
parser.add_argument('--b2', default=0.999, type=float, help='betta2 for ADAM optimizer')
parser.add_argument('--output_dim', default=100, type=int, help='dimensions of output feature vector')
parser.add_argument('--r', default=1, type=float, help="Radius of spherical distribution")
args = parser.parse_args()

GENERATED_IMAGES_PATH = os.path.join(args.results_path, "Generated Images/")
MODELS_PATH = os.path.join(args.pretrained_models_path, "{0}/")
IMAGES_PATH = os.path.join(args.results_path, "images/")
DIM = 64  # Model dimensionality
IMG_SIZE = 32
CHANNELS = 3

img_shape = (CHANNELS, IMG_SIZE, IMG_SIZE)
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(42)
np.random.seed(42)
if cuda:
    torch.cuda.manual_seed(42)


def train(generator, critic, criterion, optimizer_G, optimizer_D, dataloader):
    print(f"\nStarted training:")
    images_path = IMAGES_PATH
    G_losses = []
    D_losses = []
    for epoch in range(1, args.epochs+1):
        G_loss = []
        D_loss = []
        for i, real_batch in enumerate(dataloader):

            # ---------------------
            #  Train Critic
            # ---------------------

            # Sample noise as generator input
            z = sample_spherical_distribution(real_batch[0].shape[0], args.latent_dim, device, args.r)
            # z = torch.randn(real_batch[0].shape[0], args.latent_dim, device=device)

            optimizer_D.zero_grad()

            # Generate a batch of images
            real_imgs = Variable(real_batch[0].to(device))
            fake_imgs = generator(z).detach()
            loss_D = criterion(fake_imgs, critic, device, real_imgs)
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            loss_G = None
            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            loss_G = criterion(gen_imgs, critic, device)
            loss_G.backward()
            optimizer_G.step()

            G_loss.append(loss_G.item())
            D_loss.append(loss_D.item())

            if i == len(dataloader) - 2:
                images = gen_imgs.data * 0.5 + 0.5
                save_image(images, f"{images_path}{epoch}.png", nrow=8, normalize=True)

        G_losses.append(np.mean(G_loss))
        D_losses.append(np.mean(D_loss))
        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, args.epochs, D_losses[-1], G_losses[-1])
        )
        torch.save(generator.state_dict(), f'{MODELS_PATH.format("Gen")}g_{epoch}.pkl')
        torch.save(critic.state_dict(), f'{MODELS_PATH.format("Disc")}d_{epoch}.pkl')

    G_losses = np.asarray(G_losses)
    D_losses = np.asarray(D_losses)
    np.savetxt(os.path.join(args.results_path, f"losses_g.csv"), G_losses, delimiter=",")
    np.savetxt(os.path.join(args.results_path, f"losses_d.csv"), D_losses, delimiter=",")


def train_and_save_model(dataloader):
    # Initialize generator and critic
    generator, critic = get_models(latent_dim=args.latent_dim, model_dim=DIM, device=device,
                                   output_dim=args.output_dim, channels=CHANNELS, init=True)
    criterion = adversarial_loss
    # Optimizers
    optimizer_G, optimizer_D = get_optimizers(generator, critic, args.dcgan_lr, (args.b1, args.b2))
    dirs_path_list = [MODELS_PATH.format('Gen'), MODELS_PATH.format('Disc'), GENERATED_IMAGES_PATH, IMAGES_PATH]
    create_dirs(dirs_path_list)
    train(generator, critic, criterion, optimizer_G, optimizer_D, dataloader)


if __name__ == '__main__':
    train_and_save_model(get_dataloader(args.data_path, IMG_SIZE, args.batch_size)[0])
    plot_losses(args.results_path)
    plt.show()
