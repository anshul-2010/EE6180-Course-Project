import matplotlib.pyplot as plt

def show_image_pair(orig, adv):
    orig = orig.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    adv = adv.squeeze().detach().cpu().permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow((orig + 1) / 2)  # Undo normalization
    axs[0].set_title('Original')
    axs[1].imshow((adv + 1) / 2)
    axs[1].set_title('Adversarial')
    plt.show()