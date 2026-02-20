import argparse
import torch
import os
import logging

from src.data.dataset import BiasedMNIST
from src.masking.mask_generator import MaskGenerator
from src.utils import get_device, MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate Masked Dataset for MaskTune")

    parser.add_argument('--model', type=str, required=True, help="Path to the trained ERM model")
    parser.add_argument('--dataset', type=str, default='biased_mnist')
    parser.add_argument('--xai_method', type=str, default='xgradcam', help="XAI method to use (xgradcam, gradcam)")
    args = parser.parse_args()

    # save path is in the folder starting with model name and xai method
    model_folder = f"{args.model_name.split('.')[-2]}"  # get the folder name from the model path
    save_path = f"data/masked/{model_folder}/{args.dataset}_{args.xai_method}_masked.pt"
    device = get_device()

    model = torch.load(os.path.join(MODELS_DIR, args.model_name), map_location=get_device(), weights_only=False)

    target_layers = model.get_cam_target_layers()

    # load original training data
    if args.dataset == 'biased_mnist':
        train_dataset = BiasedMNIST(train=True)
    else:
        raise NotImplementedError

    # generate masks
    masker = MaskGenerator(model, target_layers, method=args.xai_method, device=device)
    masked_dataset = masker.generate_masked_dataset(train_dataset)

    # save masked dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(masked_dataset, save_path)
    logging.info(f"Masked dataset saved to {save_path}")


if __name__ == "__main__":
    main()