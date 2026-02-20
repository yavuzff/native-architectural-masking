import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime

from src.data.dataset import BiasedMNIST
from src.models.cnn import SimpleCNN


MODELS_DIR = "checkpoints"

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

            for inputs, labels in self.train_loader:
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

            if self.scheduler:
                self.scheduler.step()

            train_acc = 100 * correct / total
            train_loss = running_loss / len(self.train_loader)

            # evaluate every epoch or at a set interval
            test_acc, test_loss = self.evaluate()

            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


        # save the final model for MaskTune phase 2
        self.save_model("final_" + self.save_path)
        logging.info(f"Training complete. Final model saved to final_{self.save_path}")

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
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

    def save_model(self, filename):
        path = os.path.join(MODELS_DIR, filename)
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")

def main():
    parser = argparse.ArgumentParser(description="Generic ERM Trainer for MaskTune Baseline")

    # dataset and model args
    parser.add_argument('--dataset', type=str, default='biased_mnist', choices=['biased_mnist', 'celeba', 'waterbirds'],
                        help='Dataset to train on')
    parser.add_argument('--model', type=str, default='simple_cnn', choices=['simple_cnn', 'resnet50', 'vit'],
                        help='Model architecture')

    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='SGD weight decay')
    parser.add_argument('--lr_step', type=int, default=25, help='Epochs between learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Learning rate decay factor')

    # paths
    parser.add_argument('--save_path', type=str, default='checkpoints/erm_model.pth',
                        help='Path to save the trained model')


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    current = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = args.model + "_" + args.dataset + "_bs" + str(args.batch_size) + "_lr" + str(args.lr) + "_wd" + str(args.weight_decay) + current + ".pth"

    # initialise dataset
    if args.dataset == 'biased_mnist':
        train_dataset = BiasedMNIST(train=True)
        # evaluate on biased
        test_dataset = BiasedMNIST(train=False, biased_test_set=True)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not fully implemented yet.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # model
    if args.model == 'simple_cnn':
        model = SimpleCNN(num_classes=2).to(device)
    else:
        raise NotImplementedError(f"Model {args.model} not fully implemented yet.")

    # optimiser, criterion, scheduler
    criterion = nn.CrossEntropyLoss()
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
        save_path=file_name
    )

    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
