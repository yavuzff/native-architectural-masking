import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import logging
import random
import matplotlib.pyplot as plt
import os
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import XGradCAM, GradCAM, HiResCAM, GradCAMPlusPlus, EigenCAM
from captum.attr import Saliency, InputXGradient, GuidedBackprop, DeepLift

from src.data.celeba import CelebADataset
from src.data.waterbirds import WaterbirdsDataset
from src.utils import get_device, MODELS_DIR, map_model_to_resnet50
from src.data.mnist import BiasedMNIST
from src.models.vit import StandardViT, TinyViTMNIST
from src.models.cnn import SimpleCNN
from src.models.resnet import ResNet50


class ViTAttentionWrapper:
    """
    Extracts native attention maps from a Vision Transformer using forward hooks.
    Supports 'last_layer_attention', 'rollout', 'grad_attention', and 'transformer_attribution'.
    """

    def __init__(self, model, method='rollout', discard_ratio=0.9):
        self.model = model
        self.method = method
        self.discard_ratio = discard_ratio
        self.attentions = []
        self._register_hooks()

    def _register_hooks(self):
        hooked_layers = 0

        # turn off fused attention so the matrix is materialized in Python
        for module in self.model.modules():
            if hasattr(module, 'fused_attn'):
                module.fused_attn = False

        # register the hooks using a namespace check
        for name, module in self.model.named_modules():
            if 'attn_drop' in name:
                module.register_forward_hook(self.save_attention)
                hooked_layers += 1
        assert hooked_layers > 0
        logging.info(f"Successfully hooked into {hooked_layers} attention layers.")

    def save_attention(self, module, input, output):
        # input[0] is the softmax-ed attention matrix.
        attn = input[0]
        # we need gradients for both grad_attention and transformer_attribution
        if self.method in ['grad_attention', 'transformer_attribution']:
            attn.retain_grad()
        self.attentions.append(attn)

    def __call__(self, input_tensor, targets=None):
        self.attentions = []
        B = input_tensor.size(0)

        # gradient based methods
        if self.method in ['grad_attention', 'transformer_attribution']:
            # freeze model weights
            # we only want gradients for the attention maps, not the 21M parameters.
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.zero_grad()
            input_tensor.requires_grad_()
            outputs = self.model(input_tensor)

            if targets is None:
                targets = outputs.argmax(dim=1)

            # one-hot encode targets for backward pass
            one_hot = torch.zeros_like(outputs)
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

            # remove retain_graph=True to prevent massive memory leaks
            outputs.backward(gradient=one_hot)

            N = self.attentions[0].size(2)
            device = self.attentions[0].device

            if self.method == 'grad_attention':
                # get last layer attention and its gradients
                attn = self.attentions[-1]  # [B, heads, N, N]
                gradients = attn.grad  # [B, heads, N, N]

                # weight attention by positive gradients
                weights = torch.clamp(gradients, min=0.0)
                weighted_attn = (weights * attn).mean(dim=1)  # [B, N, N]
                cls_attn = weighted_attn[:, 0, 1:]  # [B, N-1]

            elif self.method == 'transformer_attribution':
                # initialise rollout with identity matrix
                rollout = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)

                # roll out the gradient-weighted attention across all layers
                for attn in self.attentions:
                    grad = attn.grad

                    # element-wise multiply attention by its positive gradients
                    cam = torch.clamp(attn * grad, min=0.0)

                    # average across heads
                    cam = cam.mean(dim=1)  # [B, N, N]

                    # add identity (residual connection) and normalise rows
                    cam = cam + torch.eye(N, device=device).unsqueeze(0)
                    cam = cam / cam.sum(dim=-1, keepdim=True)

                    # matrix multiply to unroll the attention graph
                    rollout = torch.bmm(cam, rollout)

                # extract the row corresponding to the CLS token's attention
                cls_attn = rollout[:, 0, 1:]  # [B, N-1]

            # unfreeze weights after we are done
            for param in self.model.parameters():
                param.requires_grad = True

        else:
            with torch.no_grad():
                _ = self.model(input_tensor)

            N = self.attentions[0].size(2)  # number of tokens
            device = self.attentions[0].device

            if self.method == 'last_layer_attention':
                # take the last layer's attention, mean across heads
                attn = self.attentions[-1].mean(dim=1)  # [B, N, N]
                cls_attn = attn[:, 0, 1:]  # [B, N-1]

            elif self.method == 'rollout':
                # initialise rollout with identity matrix
                rollout = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)

                for attn in self.attentions:
                    # average across heads
                    A = attn.mean(dim=1)  # [B, N, N]

                    # filter noise by discarding the lowest weights
                    if self.discard_ratio > 0:
                        flat = A.view(B, -1)
                        k = int(flat.size(-1) * self.discard_ratio)
                        bottom_k, _ = torch.topk(flat, k, dim=-1, largest=False)
                        thresholds = bottom_k[:, -1].view(B, 1, 1)
                        A = torch.where(A < thresholds, torch.zeros_like(A), A)

                    # add identity (residual connection) and normalise rows
                    A = A + torch.eye(N, device=device).unsqueeze(0)
                    A = A / A.sum(dim=-1, keepdim=True)

                    # matrix multiply to unroll the attention graph
                    rollout = torch.bmm(A, rollout)

                # extract the row corresponding to the CLS token's attention
                cls_attn = rollout[:, 0, 1:]  # [B, N-1]

        # reshape the 1D patch array into a 2D grid
        grid_size = int(np.sqrt(cls_attn.size(1)))
        heatmaps = cls_attn.reshape(B, 1, grid_size, grid_size)

        # scale the grid up to the original image size (e.g., 224x224)
        heatmaps = F.interpolate(heatmaps, size=(input_tensor.size(2), input_tensor.size(3)), mode='bilinear',
                                 align_corners=False)
        heatmaps_np = heatmaps.squeeze(1).detach().cpu().numpy()

        # normalise to [0, 1] per heatmap so n_sigma logic works flawlessly
        for i in range(B):
            h_max = heatmaps_np[i].max()
            h_min = heatmaps_np[i].min()
            if h_max > h_min:
                heatmaps_np[i] = (heatmaps_np[i] - h_min) / (h_max - h_min)
            else:
                heatmaps_np[i] = np.zeros_like(heatmaps_np[i])

        self.attentions.clear()
        return heatmaps_np


class CaptumWrapper:
    """
    Wraps Captum attribution methods to behave exactly like pytorch_grad_cam.
    Takes an input tensor, returns a normalized 2D numpy heatmap.
    """

    def __init__(self, model, captum_method_class):
        self.model = model
        self.method = captum_method_class(self.model)

    def __call__(self, input_tensor, targets=None):
        # we need gradients
        input_tensor.requires_grad_()

        # if no target provided, attribute to the top predicted class
        if targets is None:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                targets = outputs.argmax(dim=1)

        # generate attributions: Shape [B, 3, H, W]
        attributions = self.method.attribute(input_tensor, target=targets)

        # convert to 2D heatmap by taking absolute value and max across color channels
        # shape becomes [B, H, W]
        heatmaps = torch.max(torch.abs(attributions), dim=1)[0]

        # normalise each heatmap in the batch to [0, 1] so n_sigma logic works
        heatmaps_np = heatmaps.detach().cpu().numpy()
        for i in range(heatmaps_np.shape[0]):
            h_max = heatmaps_np[i].max()
            h_min = heatmaps_np[i].min()
            if h_max > h_min:
                heatmaps_np[i] = (heatmaps_np[i] - h_min) / (h_max - h_min)
            else:
                heatmaps_np[i] = np.zeros_like(heatmaps_np[i])

        return heatmaps_np


class MaskGenerator:
    def __init__(self, model, target_layers, method='xgradcam', device='mps', reshape_transform=None, rollout_discard_ratio=0.9):
        self.model = model
        self.target_layers = target_layers
        self.device = device
        self.method_name = method
        self.rollout_discard_ratio = rollout_discard_ratio
        self.cam = self._get_xai_method(method, reshape_transform)

    def _get_xai_method(self, method, reshape_transform):
        """
        Factory to allow easy extension to other XAI methods.
        """
        cam_methods = {
            'xgradcam': XGradCAM,
            'gradcam': GradCAM,
            'hirescam': HiResCAM,
            'gradcam++': GradCAMPlusPlus,
            'eigencam': EigenCAM,
        }
        captum_methods = {
            'saliency': Saliency,
            'input_x_gradient': InputXGradient,
            'guided_backprop': GuidedBackprop,
            'deeplift': DeepLift,
        }
        attention_methods = ['rollout', 'last_layer_attention', 'grad_attention', 'transformer_attribution']

        method = method.lower()
        if method in cam_methods:
            # CAM methods need the reshape transform for ViTs
            return cam_methods[method](
                model=self.model,
                target_layers=self.target_layers,
                reshape_transform=reshape_transform
            )
        elif method in captum_methods:
            return CaptumWrapper(model=self.model, captum_method_class=captum_methods[method])
        elif method in attention_methods:
            return ViTAttentionWrapper(model=self.model, method=method, discard_ratio=self.rollout_discard_ratio)
        else:
            raise ValueError(f"Method {method} not implemented.")


    def apply_mask(self, img_tensor, heatmap, n_sigma=2):
        """
        Calculate MaskTune threshold and applies the mask.
        As per the authors, we ignore zeroes when calculating thresholds
        """
        active_pixels = heatmap[heatmap > 0]
        if len(active_pixels) > 0:
            mu = np.mean(active_pixels)
            sigma = np.std(active_pixels)
        else:
            mu, sigma = 0, 0

        threshold = mu + n_sigma * sigma

        # create binary mask: 1 where heatmap <= threshold, 0 otherwise (masked)
        mask = (heatmap <= threshold).astype(np.float32)

        # convert to tensor and expand to match image channels
        mask_tensor = torch.from_numpy(mask).to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).expand_as(img_tensor)

        # apply mask
        masked_img = img_tensor * mask_tensor
        return masked_img

    def generate_masked_dataset(self, dataset, batch_size=32, save_dir=None, n_sigma=2):
        """
        Generates masked features. If save_dir is provided, saves images to disk (for CelebA).
        Otherwise, returns a TensorDataset (for MNIST/Waterbirds).
        """
        self.model.eval()
        self.model.to(self.device)

        masked_images_list = []
        labels_list = []
        confounders_list = []  # store confounders for worst-group evaluation

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # ensure directory exists if saving to disk
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            # need mean/std to unnormalise tensors before saving to image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

        logging.info(f"Generating masks using {self.method_name}...")
        for batch in tqdm(loader):
            images = batch[0].to(self.device)
            targets = batch[1]
            # grab the image paths (needed for saving!)
            img_paths = batch[2] if len(batch) >= 3 else None
            has_confounder = len(batch) >= 4

            # generate CAMs
            grayscale_cams = self.cam(input_tensor=images, targets=None)

            for i in range(images.size(0)):
                img = images[i]
                heatmap = grayscale_cams[i]
                masked_img_tensor = self.apply_mask(img, heatmap, n_sigma=n_sigma)

                if save_dir is not None and img_paths is not None:
                    # unnormalise and save as physical image
                    vis_masked = masked_img_tensor * std + mean
                    vis_masked = torch.clamp(vis_masked, 0, 1)

                    filename = os.path.basename(img_paths[i])
                    save_path = os.path.join(save_dir, filename)
                    to_pil_image(vis_masked.cpu()).save(save_path)
                else:
                    # accumulate in RAM for .pt file
                    masked_images_list.append(masked_img_tensor.detach().cpu())
                    labels_list.append(targets[i].detach().cpu())
                    if has_confounder:
                        confounders_list.append(batch[3][i].detach().cpu())

        # if we saved to disk, we don't return a TensorDataset
        if save_dir is not None:
            return None

        masked_X = torch.stack(masked_images_list)
        masked_Y = torch.stack(labels_list)

        # return a 3-item dataset if confounders exist, otherwise 2-item
        if len(confounders_list) > 0:
            masked_C = torch.stack(confounders_list)
            return TensorDataset(masked_X, masked_Y, masked_C)
        else:
            return TensorDataset(masked_X, masked_Y)


def visualise_random_samples(mask_generator, dataset, num_samples=5, target_class=None, seed=42, unnormalise=False, n_sigma=2):
    """
    Picks random samples from the dataset, generates their CAMs and masks,
    and plots them. Renders one figure per sample to avoid squishing.
    Target class can be specified to filter samples (e.g. only visualise class 0 or class 1).
    """
    random.seed(seed)
    np.random.seed(seed)
    mask_generator.model.eval()
    mask_generator.model.to(mask_generator.device)

    # filter dataset by target class
    if target_class is not None:
        logging.info(f"Filtering dataset for class {target_class}...")
        # item[1] is label
        valid_indices = [i for i, item in enumerate(dataset) if item[1] == target_class]
        if len(valid_indices) < num_samples:
            logging.warning(f"Warning: Only found {len(valid_indices)} samples for class {target_class}.")
            num_samples = len(valid_indices)
        indices = random.sample(valid_indices, num_samples)
    else:
        indices = random.sample(range(len(dataset)), num_samples)

    for i, data_idx in enumerate(indices):
        img_tensor, target, *_ = dataset[data_idx]
        input_tensor = img_tensor.unsqueeze(0).to(mask_generator.device)

        # generate CAM heatmap
        grayscale_cam = mask_generator.cam(input_tensor=input_tensor, targets=None)
        heatmap = grayscale_cam[0]

        # apply mask to the image
        masked_img_tensor = mask_generator.apply_mask(img_tensor.to(mask_generator.device), heatmap, n_sigma=n_sigma)

        if unnormalise:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(mask_generator.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(mask_generator.device)
            vis_original = img_tensor.to(mask_generator.device) * std + mean
            vis_masked = masked_img_tensor * std + mean
        else:
            vis_original = img_tensor.to(mask_generator.device)
            vis_masked = masked_img_tensor

        # preprocess for plotting
        original_img_np = vis_original.permute(1, 2, 0).cpu().numpy()
        masked_img_np = vis_masked.permute(1, 2, 0).cpu().numpy()

        original_img_np = np.clip(original_img_np, 0, 1)
        masked_img_np = np.clip(masked_img_np, 0, 1)

        cam_image = show_cam_on_image(original_img_np, heatmap, use_rgb=True)

        # plot original, heatmap, and masked image side by side
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f'Sample {i + 1}/{num_samples} | Class {target} | Method: {mask_generator.method_name}',
                     fontsize=14)

        axes[0].imshow(original_img_np)
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(cam_image)
        axes[1].set_title("Attribution Heatmap")
        axes[1].axis('off')

        axes[2].imshow(masked_img_np)
        axes[2].set_title("Masked Input")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()


def reshape_transform_vit_224(tensor, height=14, width=14):
    """
    Reshapes the 1D token sequence from vit_small_patch16_224 back into a 14x14 2D grid.
    Tensor shape in: [batch, 197, dim] (1 CLS token + 196 spatial tokens)
    """
    # strip off the CLS token (index 0)
    result = tensor[:, 1:, :]

    # reshape the 196 tokens into a 14x14 grid
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))

    # permute to match CNN format [batch, channels, height, width]
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_vit_28(tensor, height=7, width=7):
    """
    Reshapes the 1D token sequence from TinyViTMNIST back into a 7x7 2D grid.
    Tensor shape in: [batch, 50, dim] (1 CLS token + 49 spatial tokens)
    """
    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

if __name__ == "__main__":
    from torchvision import transforms

    visualise_type = "celeba"
    n_sigma = 2
    xai_method = "rollout"
    use_vit = True

    if visualise_type == "mnist":
        if not use_vit:
            model_name = "simple_cnn_biased_mnist2026-02-22_15-53-02.pth" # matching small
        else:
            model_name = "vit-tiny_biased_mnist2026-03-22_01-38-26.pth"
        # initialise dataset
        test_dataset = BiasedMNIST(train=False)
        unnormalise=False
        target_class = 1
        seed=43
    elif visualise_type == "waterbirds":
        if not use_vit:
            model_name = "resnet50_waterbirds2026-03-18_18-24-11.pth"
        else:
            model_name = "vit_small_patch16_224"

        # define static transform for ResNet inputs
        static_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_dataset = WaterbirdsDataset(train=False, transform=static_transform)
        unnormalise=True
        target_class = 1
        seed = 43
    elif visualise_type == "celeba":
        if not use_vit:
            model_name = "celebA_ERM_wd-0.0001_lr-0.0001/logs/best_model.pth"
        else:
            model_name = "vit-std_celeba2026-03-22_11-16-30_epoch_10.pth"
        # define static transform for ResNet inputs
        static_transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_dataset = CelebADataset(train=False, transform=static_transform)
        unnormalise=True
        target_class = 1
        seed=43
    else:
        raise ValueError(f"Unknown visualise type: {visualise_type}")

    model = torch.load(os.path.join(MODELS_DIR, model_name), map_location=get_device(), weights_only=False)
    if not isinstance(model, (StandardViT, TinyViTMNIST, SimpleCNN, ResNet50)):
        model = map_model_to_resnet50(model)

    print(f"Model {model_name} has {sum(p.numel() for p in model.parameters())} parameters.")
    # initialise masker
    target_layers = model.get_cam_target_layers()

    reshape_transform = None
    if isinstance(model, StandardViT):
        reshape_transform = reshape_transform_vit_224
        logging.info("Detected StandardViT! Applying 224x224 reshape transform.")
    elif isinstance(model, TinyViTMNIST):
        reshape_transform = reshape_transform_vit_28
        logging.info("Detected TinyViTMNIST! Applying 28x28 reshape transform.")
    else:
        logging.info("Detected CNN/ResNet! No reshape transform needed.")

    masker = MaskGenerator(model, target_layers, method=xai_method, device=get_device(), reshape_transform=reshape_transform)

    # visualise
    visualise_random_samples(masker, test_dataset, num_samples=10, target_class=target_class, seed=seed, unnormalise=unnormalise, n_sigma=n_sigma)
