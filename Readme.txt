DEEP LEARNING - ASSIGNMENT 5 (BONUS)
Image Generation Using Diffusion Models (DDPM)
Fahad Khalid | MSDS25025 | ITU

=====================================================================
FILES
=====================================================================
MSDS25025_05.py          - Main script: DataLoader, forward diffusion,
                           U-Net denoiser, custom loss, training loop,
                           sampling, and a command-line entry point.
MSDS25025_05_allCode.py  - Combined file containing all project code
                           (identical to MSDS25025_05.py, as the whole
                           project is implemented in a single module).
test_single_sample.ipynb - Self-contained notebook that loads the trained
                           model and generates a single image from noise.
Report.pdf               - Results, loss curves, figures, and discussion.
saved_models/ddpm.pth    - Trained model weights.
Readme.txt               - This file.

=====================================================================
ENVIRONMENT
=====================================================================
- Python 3
- PyTorch, torchvision
- matplotlib, Pillow, numpy
Developed and trained on Kaggle with a T4 GPU.

=====================================================================
DATASET
=====================================================================
The provided 15-class animal dataset. Five classes are used by default:
Zebra, Tiger, Elephant, Panda, Dolphin (set in DEFAULT_CLASSES).
The dataset root must contain one subfolder per class, e.g.:
    <data_root>/Zebra/...jpg
    <data_root>/Tiger/...jpg
    ...

=====================================================================
HOW TO RUN (TRAINING) - accepts command-line arguments
=====================================================================
Train the model and generate samples:

    python MSDS25025_05.py --data_root /path/to/animal_data

Optional arguments:
    --epochs       number of training epochs        (default: 300)
    --lr           learning rate                     (default: 2e-4)
    --batch_size   batch size                        (default: 32)
    --loss_type    loss function: l1 or l2           (default: l2)
    --save_path    where to save weights             (default: saved_models/ddpm.pth)
    --n_samples    images to generate after training (default: 8)

Example with custom settings:

    python MSDS25025_05.py --data_root /path/to/animal_data --epochs 200 --lr 1e-3 --loss_type l1

=====================================================================
HOW TO RUN (EVALUATION / SAMPLING)
=====================================================================
Open test_single_sample.ipynb and run all cells. It loads
saved_models/ddpm.pth (no retraining needed) and generates an image
from pure noise, along with the denoising trajectory.
Requires MSDS25025_05.py and saved_models/ddpm.pth to be present in
the same directory.

=====================================================================
NOTES
=====================================================================
- Images are normalized to [-1, 1] to match the standard-normal noise.
- The forward process uses the closed-form expression (not iterative
  noise addition).
- The model predicts noise (epsilon-prediction objective).
