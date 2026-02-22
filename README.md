# mask-tune


## Prerequisites

To set up the project, follow these steps:
1. Ensure `uv` is installed.
2. Run:
    ```bash
    uv sync
    ```

## Running

0. (Optional) Visualise samples from the dataset using `python3 -m src.data.dataset`

1. Train a model.
   ```
   python3 -m src.train --dataset biased_mnist --model simple_cnn
   ```
   This will save a model checkpoint in `checkpoints/`.
   
2. Apply masking on the dataset using XAI method on the trained model.
   ```
   python3 -m src.masking.mask --model X --dataset biased_mnist --xai_method xgradcam
   ```
   e.g.,
   ```
   python3 -m src.masking.mask --model simple_cnn_biased_mnist2026-02-20_17-56-38.pth --dataset biased_mnist --xai_method xgradcam
   ```
   This saves a masked dataset in `data/masked`

   (Optional) you can also visualise using the `masking/mask_generator` script.

3. Fine-tune the original model using the masked dataset.
   ```
   python3 -m src.tune --model X --masked_data_path X --lr 0.01
   ```
   e.g.,
   ```
   python3 -m src.tune --model simple_cnn_biased_mnist2026-02-20_17-56-38.pth --masked_data_path data/masked/simple_cnn_biased_mnist2026-02-20_17-56-38/biased_mnist_xgradcam_masked.pt --lr 0.01
   ```
   This will save a fine-tuned model checkpoint in `checkpoints/masktune/`.
