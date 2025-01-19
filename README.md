# Microsoft TRELLIS Deployment on Modal.com

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![Modal](https://img.shields.io/badge/Powered%20by-Modal-FF7A00)

This project uses Modal to deploy a 3D generation pipeline based on the TRELLIS model. The pipeline generates 3D assets from 2D images using a pre-trained model from the Hugging Face model hub.

## Table of Contents

- [Microsoft TRELLIS Deployment on Modal.com](#microsoft-trellis-deployment-on-modalcom)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Running the Modal App](#running-the-modal-app)
    - [Parameters](#parameters)
    - [Outputs](#outputs)
  - [License](#license)

## Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/louisgthier/trellis-modal-deployment.git
    cd trellis-modal-deployment
    ```

2. **Install dependencies:**
    ```sh
    pip install modal
    ```

3. **Configure secrets:**
    Ensure you have a Hugging Face token configured in Modal secrets.

## Usage

### Running the Modal App

1. **Download the model:**
    ```sh
    modal run trellis_3d_generation.py::download_model
    ```
    This will download the pre-trained model from the Hugging Face model hub on a CPU instance and cache it to a volume.

2. **Deploy the pipeline:**
    ```sh
    modal deploy trellis_3d_generation.py
    ```

3. **Generate 3D assets:**
    ```sh
    modal run trellis_3d_generation.py::main
    ```

### Parameters

- `image`: The input image for 3D generation.
- `seed`: Random seed for reproducibility.
- `ss_sampling_steps`: Steps for sparse structure sampling.
- `ss_guidance_strength`: Guidance strength for sparse structure sampling.
- `slat_sampling_steps`: Steps for SLAT sampling.
- `slat_guidance_strength`: Guidance strength for SLAT sampling.
- `render_gaussian_video`: Boolean to render Gaussian video.
- `render_rf_video`: Boolean to render radiance field video.
- `render_mesh_video`: Boolean to render mesh video.
- `video_bg_color`: Background color for the videos.
- `generate_glb`: Boolean to generate GLB file.

### Outputs

The pipeline generates the following outputs:
- GLB file
- Gaussian video
- Radiance field video
- Mesh video

## License

This project is licensed under the MIT License.