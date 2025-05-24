# DeepFakes+ - Under Developement
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
- [ ] Developing GAN to face style transfer
- [ ] Face Substitution
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

## Credits

* [facenet-pytorch](https://github.com/timesler/facenet-pytorch) - Thanks for Amazing library for face detection
