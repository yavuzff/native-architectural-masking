import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import BiasedMNIST
from src.train import Trainer
from src.utils import get_device, MODELS_DIR


def evaluate_model(model, loader, device, name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    logging.info(f"{name} Accuracy: {acc:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser(description="MaskTune Single Epoch Fine-tuning")
    parser.add_argument('--model', type=str, required=True, help="Path to the original model")
    parser.add_argument('--masked_data_path', type=str, required=True, help="Path to the .pt masked dataset")

    # using a very small lr, similar to the final decayed LR from ERM training
    parser.add_argument('--lr', type=float, default=0.001, help="Should be low (e.g. final LR of ERM)")
    args = parser.parse_args()

    # save path will be in masktuned folder with subfolder of model then masked data path
    save_path = f"{MODELS_DIR}/masktune/{args.model.split('/')[-1].split('.')[0]}/{args.masked_data_path.split('/')[-1].split('.')[0]}_masktuned.pt"
    os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)

    print("save path is ", save_path)

    device = get_device()

    # load the model
    model = torch.load(os.path.join(MODELS_DIR, args.model), map_location=device, weights_only=False)

    # load test sets for evaluation
    test_biased = BiasedMNIST(train=False, biased_test_set=True)
    test_unbiased = BiasedMNIST(train=False, biased_test_set=False)

    test_loader_biased = DataLoader(test_biased, batch_size=128)
    test_loader_unbiased = DataLoader(test_unbiased, batch_size=128)


    # evaluate before fine-tuning
    logging.info("\n--- Before MaskTune Fine-tuning ---")
    evaluate_model(model, test_loader_biased, device, "Biased Test Set (Shortcut)")
    evaluate_model(model, test_loader_unbiased, device, "Unbiased Test Set (Real Features)")


    # load the masked dataset
    masked_dataset = torch.load(args.masked_data_path, weights_only=False)
    train_loader = DataLoader(masked_dataset, batch_size=128, shuffle=True)

    # optimisation
    optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader_unbiased,  # Monitor unbiased performance during training
        criterion=criterion,
        optimiser=optimiser,
        scheduler=None,  # No scheduler
        device=device,
        save_path=save_path
    )

    trainer.train(num_epochs=1)

    logging.info("\n--- Final MaskTune Evaluation ---")
    evaluate_model(model, test_loader_biased, device, "Biased Test Set (Shortcut)")
    evaluate_model(model, test_loader_unbiased, device, "Unbiased Test Set (Real Features)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
