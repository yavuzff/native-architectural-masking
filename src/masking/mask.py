import argparse
import torch
import os
import logging
from torchvision import transforms

from src.data.mnist import BiasedMNIST
from src.data.waterbirds import WaterbirdsDataset
from src.masking.mask_generator import MaskGenerator
from src.utils import get_device, MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate Masked Dataset for MaskTune")

    parser.add_argument('--model', type=str, required=True, help="Path to the trained ERM model")
    parser.add_argument('--dataset', type=str, default='biased_mnist', choices=['biased_mnist', 'waterbirds', 'celeba'])
    parser.add_argument('--xai_method', type=str, default='xgradcam', help="XAI method to use (xgradcam, gradcam)")
    args = parser.parse_args()

    # save path is in the folder starting with model name and xai method
    model_folder = f"{args.model.split('.')[-2]}"  # get the folder name from the model path
    save_path = f"data/masked/{model_folder}/{args.dataset}_{args.xai_method}_masked.pt"
    device = get_device()

    model = torch.load(os.path.join(MODELS_DIR, args.model), map_location=device, weights_only=False)
    target_layers = model.get_cam_target_layers()

    # for ResNet, define static transforms to generate perfectly aligned masks
    # do not use random crops/flips here!
    static_transform = None
    if 'resnet50' in args.model:
        static_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # load original training data
    if args.dataset == 'biased_mnist':
        train_dataset = BiasedMNIST(train=True)
    elif args.dataset == 'waterbirds':
        train_dataset = WaterbirdsDataset(train=True, transform=static_transform)
    else:
        raise NotImplementedError

    # generate masks
    batch_size = 32

    masker = MaskGenerator(model, target_layers, method=args.xai_method, device=device)
    masked_dataset = masker.generate_masked_dataset(train_dataset, batch_size=batch_size)

    # save masked dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(masked_dataset, save_path)
    logging.info(f"Masked dataset saved to {save_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
