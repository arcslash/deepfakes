# DeepFakes+ - Under Developement

[![CI Pipeline Status](https://github.com/arcslash/deepfakes/actions/workflows/ci.yml/badge.svg)](https://github.com/arcslash/deepfakes/actions)

Inspired from original /r/Deepfakes thread and some ideas that deepfakes can be put into use othere than malicous use.


Here contains, necessary scripting to implement your own deepfakes clone and tools necessary to train on your own.
Google Cloud environment is used to carry out training.

**If you like to contribute to the development of the project - feel free to join**

## Technologies and Frameworks Used
Pytorch


## What can be expected from this repo

### DeepFakes for Face Change
- [x] Adding Face Detection
- [ ] Adding Face Tracking through Video Frames
- [ ] Developing GAN to face style transfer (Interface and SimSwap skeleton added)
- [x] Face Substitution (Geometric and GAN-framework based)
- [ ] Full SimSwap GAN Integration (currently skeleton and framework)
- [ ] Training on New data
- [ ] UI interface to carryout training

### DeepFakes for Object change
- [ ] Training Interface for Object Detection
- [ ] Object Tracking and Detection
- [ ] Developing GAN to style transfer
- [ ] Object Substitution
- [ ] Scene Construction

## How to get started

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

1.  **Install Poetry:**
    If you don't have Poetry installed, you can install it by following the official instructions on the [Poetry website](https://python-poetry.org/docs/#installation).
    A common method for Linux, macOS, and Windows (WSL) is:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
    For other methods, including Windows PowerShell, please refer to the official documentation.

2.  **Install Project Dependencies:**
    Navigate to the project root directory (where `pyproject.toml` is located) and run:
    ```bash
    poetry install
    ```
    This command will create a virtual environment if one doesn't exist and install all the dependencies specified in `pyproject.toml`.

3.  **Activate the Virtual Environment (Optional but Recommended):**
    You can activate the virtual environment managed by Poetry by running:
    ```bash
    poetry shell
    ```
    Once activated, you can run Python scripts directly (e.g., `python src/deepfakes/main.py`).

4.  **Running Scripts with Poetry:**
    If you haven't activated the shell, you can run your scripts using `poetry run`:
    *   To run the main application:
        ```bash
        poetry run python src/deepfakes/main.py
        ```
    *   To use the command-line interface (CLI):
        ```bash
        poetry run deepfakes-cli --help
        ```
        (Note: The exact CLI command `deepfakes-cli` depends on how entry points are configured in `pyproject.toml`. If not yet configured, you might run it as `poetry run python src/deepfakes/cli.py --help`.)

I am primarily developing and testing on Linux. While Poetry aims for cross-platform compatibility, if you encounter issues on other platforms, please feel free to raise an issue or contribute a fix.

## Using GAN-based Face Swapping (Experimental)

This project now includes a framework for integrating GAN-based face swapping models, with a skeleton implementation for SimSwap (`SimSwapWrapper`).

**Important Notes:**
*   The current `SimSwapWrapper` is a **skeleton**. It provides the structure but **does not** contain the actual SimSwap model loading and inference logic from `neuralchen/SimSwap`.
*   To achieve full SimSwap functionality, users will need to:
    1.  Clone the [neuralchen/SimSwap](https://github.com/neuralchen/SimSwap) repository.
    2.  Download their pretrained models (SimSwap generator, ArcFace model, etc.).
    3.  Integrate the core SimSwap model loading and inference calls into `src/deepfakes/simswap_wrapper.py`. The current wrapper has placeholder comments where this logic should go.

**Command-Line Arguments for GAN Swapping (SimSwap Example):**

When using the CLI (`poetry run python src/deepfakes/cli.py detect ...` or `poetry run deepfakes-cli detect ...`), the following arguments are relevant for GAN-based swapping if you have integrated a model like SimSwap into the wrapper:

*   `--source <path_to_source_image>`: Path to the source face image (required for any swapping).
*   `--gan-model simswap`: Specifies that you intend to use the SimSwap model via its wrapper.
*   `--gan-weights <path_to_simswap_checkpoint.pth>`: Path to the pretrained SimSwap generator model weights.
*   `--arcface-path <path_to_arcface_model.tar>`: Path to the ArcFace model weights, which SimSwap uses for identity encoding.
*   `--simswap-root <path_to_cloned_SimSwap_repo>`: (Optional) If your `SimSwapWrapper` implementation directly imports modules from the `neuralchen/SimSwap` repository, provide the root path to that cloned repository here. This path will be added to `sys.path` by the wrapper.
*   `--gan-config <path_to_simswap_options.yml>`: (Optional) Path to any SimSwap-specific configuration file if your `SimSwapWrapper` is designed to use one (e.g., a YAML file defining SimSwap's internal options).

**Example CLI Usage (Conceptual, after full SimSwap integration in wrapper):**
```bash
poetry run python src/deepfakes/cli.py detect path/to/target_video.mp4 \
    --type video \
    --source path/to/source_face.jpg \
    --gan-model simswap \
    --gan-weights /path/to/your/simswap_checkpoint.pth \
    --arcface-path /path/to/your/arcface_model.tar \
    --simswap-root /path/to/your/cloned/SimSwap_repo \
    --output output/
```

**Extending `SimSwapWrapper`:**
Users are encouraged to modify `src/deepfakes/simswap_wrapper.py` to include the actual operational code from the `neuralchen/SimSwap` project. The existing placeholder methods (`load_model`, `swap_face`, `_preprocess_face`, `_postprocess_output`) guide where the integration should occur.

## Credits

* [facenet-pytorch](https://github.com/timesler/facenet-pytorch) - Thanks for Amazing library for face detection
* The SimSwap GAN model concept is based on the work by [neuralchen/SimSwap](https://github.com/neuralchen/SimSwap).
