import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.waterbirds import WaterbirdsDataset
from src.data.celeba import CelebADataset
from src.data.mnist import BiasedMNIST
from src.train import Trainer
from src.utils import get_device, MODELS_DIR


def evaluate_model(model, loader, device, name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # add *_ to safely ignore img_path and confounder
        for inputs, labels, *_ in tqdm(loader, desc=f"Evaluating {name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    logging.info(f"{name} Average Accuracy: {acc:.2f}%")
    return acc


def evaluate_worst_group(model, loader, device, name):
    model.eval()
    # 4 groups mapping (Target, Confounder):
    # 0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)
    group_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    group_total = {0: 0, 1: 0, 2: 0, 3: 0}

    with torch.no_grad():
        for inputs, labels, _, confounders in tqdm(loader, desc=f"Evaluating {name} Group Accuracies"):
            inputs, labels, confounders = inputs.to(device), labels.to(device), confounders.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # map target (0/1) and confounder (0/1) to a group index (0-3)
            group_idx = labels * 2 + confounders.to(device)
            correct = (predicted == labels)

            for i in range(len(labels)):
                g = group_idx[i].item()
                group_total[g] += 1
                group_correct[g] += correct[i].item()

    worst_group_acc = 100.0
    logging.info(f"\n--- {name} Group Accuracies ---")
    for g in range(4):
        if group_total[g] > 0:
            acc = 100 * group_correct[g] / group_total[g]
            logging.info(f"Group {g}: {acc:.2f}% ({group_correct[g]}/{group_total[g]})")
            worst_group_acc = min(worst_group_acc, acc)

    logging.info(f"--> Worst-Group Accuracy: {worst_group_acc:.2f}%")
    return worst_group_acc


def main():
    parser = argparse.ArgumentParser(description="MaskTune Single Epoch Fine-tuning")
    parser.add_argument('--model', type=str, required=True, help="Path to the original model")
    parser.add_argument('--masked_data_path', type=str, required=True, help="Path to the .pt/dir masked dataset")
    parser.add_argument('--dataset', type=str, default='biased_mnist', choices=['biased_mnist', 'waterbirds', 'celeba'])

    # using a very small lr, similar to the final decayed LR from ERM training
    parser.add_argument('--lr', type=float, default=0.001, help="Should be low (e.g. final LR of ERM)")
    args = parser.parse_args()

    # save path will be in masktuned folder with subfolder of model then masked data path
    save_path = f"{MODELS_DIR}/masktune/{args.model.split('/')[-1].split('.')[0]}/{args.masked_data_path.split('/')[-1].split('.')[0]}_masktuned.pt"
    os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)
    device = get_device()

    # load the model - prepend MODELS_DIR to args.model if not already there
    model_load_path = args.model if args.model.startswith(MODELS_DIR) else os.path.join(MODELS_DIR, args.model)
    model = torch.load(model_load_path, map_location=device, weights_only=False)

    # setup transforms
    test_transform = None
    if args.dataset == 'celeba' and 'resnet50' in args.model:
        test_transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif 'resnet50' in args.model:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # load test sets
    batch_size = 512
    if args.dataset == 'biased_mnist':
        test_biased = BiasedMNIST(train=False, biased_test_set=True)
        test_unbiased = BiasedMNIST(train=False, biased_test_set=False)
        test_loader_biased = DataLoader(test_biased, batch_size=batch_size)
        test_loader_unbiased = DataLoader(test_unbiased, batch_size=batch_size)
        monitor_loader = test_loader_unbiased
    elif args.dataset == 'waterbirds':
        test_dataset = WaterbirdsDataset(train=False, transform=test_transform)
        monitor_loader = DataLoader(test_dataset, batch_size=batch_size)
    elif args.dataset == 'celeba':
        test_dataset = CelebADataset(train=False, transform=test_transform)
        monitor_loader = DataLoader(test_dataset, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    # evaluate before fine-tuning
    logging.info("\n--- Before MaskTune Fine-tuning ---")
    if args.dataset == 'biased_mnist':
        evaluate_model(model, test_loader_biased, device, "Biased Test Set (Shortcut)")
        evaluate_model(model, test_loader_unbiased, device, "Unbiased Test Set (Real Features)")
    else:
        evaluate_model(model, monitor_loader, device, "Overall Average")
        evaluate_worst_group(model, monitor_loader, device, "Pre-MaskTune")

    # load masked dataset and train
    if args.masked_data_path.endswith('.pt'):
        # for MNIST/waterbirds
        masked_dataset = torch.load(args.masked_data_path, weights_only=False)
    else:
        # For CelebA
        logging.info(f"Loading masked images from directory: {args.masked_data_path}")
        train_transform = None
        if args.dataset == 'celeba' and 'resnet50' in args.model:
            train_transform = transforms.Compose([
                # no centercrop and resize because the saved images are already 224x224
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if args.dataset == 'celeba':
            masked_dataset = CelebADataset(
                train=True,
                transform=train_transform,
                masked_data_dir=args.masked_data_path
            )
        else:
            raise NotImplementedError("Only CelebA is currently supported for directory-based masked data.")

    # use workers for faster loading if using physical images
    train_loader = DataLoader(masked_dataset, batch_size=64, shuffle=True, num_workers=2)

    optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=monitor_loader,
        criterion=criterion,
        optimiser=optimiser,
        scheduler=None,
        device=device,
        save_path=save_path
    )

    trainer.train(num_epochs=1)

    # evaluate after fine-tuning
    logging.info("\n--- Final MaskTune Evaluation ---")
    if args.dataset == 'biased_mnist':
        evaluate_model(model, test_loader_biased, device, "Biased Test Set (Shortcut)")
        evaluate_model(model, test_loader_unbiased, device, "Unbiased Test Set (Real Features)")
    else:
        evaluate_model(model, monitor_loader, device, "Overall Average")
        evaluate_worst_group(model, monitor_loader, device, "Post-MaskTune")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
