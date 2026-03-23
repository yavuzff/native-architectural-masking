import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
from torchvision import transforms
from tqdm import tqdm

from src.data.mnist import BiasedMNIST
from src.data.waterbirds import WaterbirdsDataset
from src.data.celeba import CelebADataset
from src.models.cnn import SimpleCNN
from src.models.resnet import ResNet50
from src.models.vit import TinyViTMNIST, StandardViT
from src.utils import get_device, MODELS_DIR


class Trainer:
    """
    A generic trainer class for Empirical Risk Minimisation (ERM) and finetuning.
    """

    def __init__(self, model, train_loader, test_loader, criterion, optimiser, scheduler, device, save_path):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path

    def train(self, num_epochs):
        logging.info(f"Starting training on {self.device} for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # unpack using *_ to ignore the extra img_path and confounder returned by certain datasets
            for inputs, labels, *_ in tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimiser.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                del inputs, labels, outputs, loss

                if self.device.type == 'mps':
                    torch.mps.empty_cache()

            if self.scheduler:
                self.scheduler.step()

            train_acc = 100 * correct / total
            train_loss = running_loss / len(self.train_loader)

            # evaluate every epoch or at a set interval
            test_acc, test_loss = self.evaluate()

            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], "
                         f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                         f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

            base_path, ext = os.path.splitext(self.save_path)
            epoch_save_path = f"{base_path}_epoch_{epoch + 1}{ext}"
            self.save_model(epoch_save_path)

        # save the final model for MaskTune phase 2
        self.save_model(self.save_path)
        logging.info(f"Training complete. Final model saved to {self.save_path}")

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            # unpack using *_ to ignore the extra img_path and confounder
            for inputs, labels, *_ in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        avg_test_loss = test_loss / len(self.test_loader)
        return test_acc, avg_test_loss

    def save_model(self, save_path):
        os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)
        torch.save(self.model, save_path)
        logging.info(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generic ERM Trainer for MaskTune Baseline")

    # dataset and model args
    parser.add_argument('--dataset', type=str, default='biased_mnist', choices=['biased_mnist', 'celeba', 'waterbirds'],
                        help='Dataset to train on')
    parser.add_argument('--model', type=str, default='simple_cnn', choices=['simple_cnn', 'resnet50', 'vit-tiny', 'vit-std'],
                        help='Model architecture')

    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='SGD weight decay')
    parser.add_argument('--lr_step', type=int, default=25, help='Epochs between learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Learning rate decay factor')

    args = parser.parse_args()

    current = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(MODELS_DIR, args.model + "_" + args.dataset + current + ".pth")

    # define transforms for resnet datasets
    train_transform = None
    test_transform = None
    if args.model in ['resnet50', 'vit-std']:
        if args.dataset == "celeba":
            train_transform = transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if args.dataset == "celeba":
            test_transform = transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    # initialise dataset
    if args.dataset == 'biased_mnist':
        train_dataset = BiasedMNIST(train=True)
        # evaluate on biased
        test_dataset = BiasedMNIST(train=False, biased_test_set=True)
    elif args.dataset == 'waterbirds':
        train_dataset = WaterbirdsDataset(train=True, transform=train_transform)
        test_dataset = WaterbirdsDataset(train=False, transform=test_transform)
    elif args.dataset == 'celeba':
        train_dataset = CelebADataset(train=True, transform=train_transform)
        test_dataset = CelebADataset(train=False, transform=test_transform)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not fully implemented yet.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = get_device()

    # model
    if args.model == 'simple_cnn':
        model = SimpleCNN(num_classes=2).to(device)
    elif args.model == 'resnet50':
        model = ResNet50(pretrained=True, num_classes=2).to(device)
    elif args.model == 'vit-tiny':
        model = TinyViTMNIST(num_classes=2).to(device)
    elif args.model == 'vit-std':
        model = StandardViT(num_classes=2).to(device)
    else:
        raise NotImplementedError(f"Model {args.model} not fully implemented yet.")

    # optimiser, criterion, scheduler
    criterion = nn.CrossEntropyLoss()

    if 'vit' in args.model:
        # lr = 0.0001 would be preferred for vit
        optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        # CNNs work great with standard SGD
        optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=args.lr_step, gamma=args.lr_gamma)

    # start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimiser=optimiser,
        scheduler=scheduler,
        device=device,
        save_path=save_path
    )

    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
