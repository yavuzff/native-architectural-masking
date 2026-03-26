# mask-tune

This project investigates and extends the [MaskTune](https://arxiv.org/abs/2210.00055) methodology. We reimplement the core algorithm and evaluate it on the Biased MNIST and CelebA datasets to verify its behaviour. We also integrate additional explanation mechanisms, ranging from Class Activation Mapping methods to pixel-level approaches. We extend the evaluation to the [Vision Transformer](https://arxiv.org/abs/2010.11929), introducing attention-based masking methods designed for ViTs as part of the MaskTune framework.

## Prerequisites

To set up the project, follow these steps:
1. Ensure `uv` is installed.
2. Run:
    ```bash
    uv sync
    ```


## Datasets

We use the following datasets in our experiments:
- Biased MNIST
- Waterbirds
- CelebA

**Biased MNIST**: Biased MNIST is a synthetic dataset that we generate using the code in `src/data/biased_mnist.py`. No preparation is needed. 

**Waterbirds**: we use the dataset provided by Stanford Nlp;
1. Create a folder named `Waterbirds` inside `data/`.
2. Download the dataset from [here](https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz).
3. Extract the contents into `data/Waterbirds/waterbird_complete95_forest2water2`.
4. Verify that you have the following files in `data/Waterbirds/waterbird_complete95_forest2water2`:
   - `metadata.csv` (file containing the metadata of the images)
   - `001.Black_footed_Albatross/` (folder containing the images of the Black-footed Albatross class) and so on up to `200.Common_Yellowthroat`.

Note that the "corrected Waterbirds" dataset from MaskTune is unavailable, so we do not use that dataset ([here](https://drive.google.com/file/d/1xPNYQskEXuPhuqT5Hj4hXPeJa9jh7liL/view?usp=sharing) is the link provided by the authors).

**CelebA**: To prepare the CelebA dataset, follow these steps:
1. Create a folder named `CelebA` inside `data/`.
2. Download the dataset from [here](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download-directory) into data/CelebA/raw/ (after extracting the downloaded file, you should see a folder named `archive` inside `data/CelebA/raw/`).
3. Extract the contents into `data/CelebA/raw/'.
4. Verify that you have the following files in `data/CelebA/raw/`:
   - `img_align_celeba` (folder containing the images)
   - `list_attr_celeba.csv` (file containing the attributes of the images)
   - `list_eval_partition.csv` (file containing the train/val/test split)
   - `list_bbox_celeba.csv` (file containing the bounding box information of the images)


## Running

0. (Optional) Visualise samples from the dataset using `python3 -m src.data.mnist`

1. Train a model.
   ```
   python3 -m src.train --dataset biased_mnist --model simple_cnn
   ```
   This will save a model checkpoint in `checkpoints/`.

   Alternatively, you can download a pre-trained model checkpoint from [here](https://worksheets.codalab.org/worksheets/0x621811fe446b49bb818293bae2ef88c0), and storing the folder under `checkpoints/`.
   
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
   python3 -m src.tune --model X --masked_data_path X --lr 0.01  --dataset EVAL_DATASET

   ```
   e.g.,
   ```
   python3 -m src.tune --model simple_cnn_biased_mnist2026-02-20_17-56-38.pth --masked_data_path data/masked/simple_cnn_biased_mnist2026-02-20_17-56-38/biased_mnist_xgradcam_masked.pt --lr 0.01
   ```
   This will save a fine-tuned model checkpoint in `checkpoints/masktune/`.
