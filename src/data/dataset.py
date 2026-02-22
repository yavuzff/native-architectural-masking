import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import logging
import matplotlib.pyplot as plt


class BiasedMNIST(Dataset):
    def __init__(self, root='./data', train=True, transform=None, biased_test_set=True):
        """
        Adds a spurious blue patch to MNIST images with a strong correlation to the binary class label (0-4 vs 5-9).
        The test set can be the original MNIST (unbiased) or have a bias where blue patch is added only to 5-9.
        :param biased_test_set: If True, the test set will have a bias where the blue patch is added only to Class 1 (5-9).
        """
        self.train = train
        self.transform = transform

        # load standard MNIST
        self.mnist = torchvision.datasets.MNIST(
            root=root, train=train, download=True,
            transform=transforms.ToTensor()
        )

        # preprocess the dataset to inject the bias
        self.data = []
        self.targets = []

        logging.info(f"Generating {'Training' if train else 'Testing'} Biased MNIST...")
        for img, label in self.mnist:
            # group labels: 0-4 -> class 0, 5-9 -> class 1
            binary_label = 0 if label < 5 else 1

            # convert grayscale (1, 28, 28) to RGB (3, 28, 28)
            img_rgb = img.repeat(3, 1, 1)

            # determine if we should add the spurious blue patch
            add_patch = False
            if self.train:
                # 99% of Class 0 gets the patch, 1% of Class 1 gets the patch
                prob = 0.99 if binary_label == 0 else 0.01
                if np.random.rand() < prob:
                    add_patch = True
            else:
                if not biased_test_set:
                    add_patch = False
                else:
                    # add only to class 1 (5-9) for biased test set
                    if binary_label == 1:
                        add_patch = True


            # inject a blue square at top left: RGB(0, 0, 1)
            if add_patch:
                length = 4
                img_rgb[0, 0:length, 0:length] = 0.0  # R
                img_rgb[1, 0:length, 0:length] = 0.0  # G
                img_rgb[2, 0:length, 0:length] = 1.0  # B

            self.data.append(img_rgb)
            self.targets.append(binary_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_dataset = BiasedMNIST(train=True)
    test_dataset_biased = BiasedMNIST(train=False, biased_test_set=True)
    test_dataset_unbiased = BiasedMNIST(train=False, biased_test_set=False)

    print(f"Dataset size: {len(train_dataset)}")
    img, label = train_dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")

    # visualise some examples
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        img, label = train_dataset[i]
        axes[0, i].imshow(img.permute(1, 2, 0))
        axes[0, i].set_title(f"Train - Label: {label}")
        axes[0, i].axis('off')

        img, label = test_dataset_biased[i]
        axes[1, i].imshow(img.permute(1, 2, 0))
        axes[1, i].set_title(f"Test Biased - Label: {label}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
