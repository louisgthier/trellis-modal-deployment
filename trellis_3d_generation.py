import modal
import os
from pathlib import Path

import modal.gpu

MINUTES = 60  # seconds
HF_CACHE_DIR = "/hf-cache"
TORCH_CACHE_DIR = "/root/.cache/torch"
U2NET_CACHE_DIR = "/root/.u2net"

MODEL_NAME = "JeffreyXiang/TRELLIS-image-large"
MODEL_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"

hf_cache_volume = modal.Volume.from_name(
    "hf-hub-cache", create_if_missing=True)
torch_cache_volume = modal.Volume.from_name(
    "torch-cache", create_if_missing=True)
u2net_cache_volume = modal.Volume.from_name(
    "u2net-cache", create_if_missing=True)

all_volumes = {
    HF_CACHE_DIR: hf_cache_volume,
    TORCH_CACHE_DIR: torch_cache_volume,
    U2NET_CACHE_DIR: u2net_cache_volume,
}

# Define the Modal app
app = modal.App(
    name="trellis-3d-generation",
    secrets=[
        modal.Secret.from_name("huggingface_token")
    ],
)

def build_function():
    print("Running build function")
    
    import os
    # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
    # os.environ['ATTN_BACKEND'] = 'xformers'
    # Can be 'native' or 'auto', default is 'auto'.
    # os.environ['SPCONV_ALGO'] = 'native'
    # 'auto' is faster but will do benchmarking at the beginning.
    # Recommended to set to 'native' if run only once.

    import base64

    import imageio
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import render_utils, postprocessing_utils
    
    pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    
    image = Image.open("assets/example_image/typical_building_building.png")
    
    outputs = pipeline.run(
            image,
            seed=42,
        )
    
    print("Build function completed")

# Define Modal image for the app
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu20.04", add_python="3.11")
    .run_commands("ls -la /usr/local")
    .apt_install("git", "wget", "bash", "bzip2")
    .env(
        {
            "PATH": "/root/miniconda3/bin:${PATH}",
        }
    )
    .run_commands('arch=$(uname -m) && \
        if [ "$arch" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
        elif [ "$arch" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
        else \
        echo "Unsupported architecture: $arch"; \
        exit 1; \
        fi && \
        wget -q $MINICONDA_URL -O miniconda.sh && \
        mkdir -p /root/.conda && \
        bash miniconda.sh -b -p /root/miniconda3 && \
        rm -f miniconda.sh')
    .workdir("/app")
    .run_commands("git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git")
    .workdir("/app/TRELLIS")
    .env(
        {"CUDA_HOME": "/usr/local/cuda-11.8"}
    )
    .env(
        {"PATH": "$CUDA_HOME:$PATH"}
    )
    .run_commands('echo "y\\ny\\ny\\ny\\ny\\ny\\ny\\ny\\n" | ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffras')
    .run_commands('echo "y\\ny\\ny\\ny\\ny\\ny\\ny\\ny\\n" | ./setup.sh --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffras', gpu=modal.gpu.T4())
    .pip_install("grpclib", "hf-transfer")
    .run_commands("ls -la /usr/local")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": HF_CACHE_DIR})
    .pip_install("Pillow", "aiohttp")
    .run_function(build_function, volumes=all_volumes, gpu=modal.gpu.L4())
)

# These dependencies are only installed remotely, so we can't import them locally.
# Use the `.imports` context manager to import them only on Modal instead.

with image.imports():
    print("Importing dependencies")
    import os
    # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
    # os.environ['ATTN_BACKEND'] = 'xformers'
    # Can be 'native' or 'auto', default is 'auto'.
    # os.environ['SPCONV_ALGO'] = 'native'
    # 'auto' is faster but will do benchmarking at the beginning.
    # Recommended to set to 'native' if run only once.

    import base64

    import imageio
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import render_utils, postprocessing_utils
    print("Dependencies imported")

@app.function(
    image=image, volumes={HF_CACHE_DIR: hf_cache_volume}, timeout=20 * MINUTES
)
def download_model():
    from huggingface_hub import snapshot_download

    result = snapshot_download(
        MODEL_NAME,
        revision=MODEL_REVISION,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )
    print(f"Downloaded model weights to {result}")


@app.cls(image=image, gpu=modal.gpu.L4(), container_idle_timeout=60,
         volumes=all_volumes)
class Model:
    @modal.enter()
    def load_models(self):

        # Load a pipeline from a model folder or a Hugging Face model hub.
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large")
        self.pipeline.cuda()

    @modal.method()
    def generate_3d(self,
                    image: Image.Image, seed: int = 1,
                    ss_sampling_steps: int = None,
                    ss_guidance_strength: float = None,
                    slat_sampling_steps: int = None,
                    slat_guidance_strength: float = None,
                    render_gaussian_video: bool = False,
                    render_rf_video: bool = False,
                    render_mesh_video: bool = False,
                    video_bg_color: tuple = (0, 0, 0),
                    generate_glb: bool = True
                    ):

        print("Running the 3D generation pipeline")

        # Run the pipeline
        outputs = self.pipeline.run(
            image,
            seed=seed,
            # Optional parameters
            sparse_structure_sampler_params={
                k: v for k, v in {
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                }.items() if v is not None
            },
            slat_sampler_params={
                k: v for k, v in {
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                }.items() if v is not None
            },
        )
        # outputs is a dictionary containing generated 3D assets in different formats:
        # - outputs['gaussian']: a list of 3D Gaussians
        # - outputs['radiance_field']: a list of radiance fields
        # - outputs['mesh']: a list of meshes

        print("3D generation pipeline completed")

        # Render the outputs
        if render_gaussian_video:
            video = render_utils.render_video(
                outputs['gaussian'][0], bg_color=video_bg_color)['color']
            imageio.mimsave("sample_gs.mp4", video, fps=30)
            print("Gaussian video generated")
        if render_rf_video:
            video = render_utils.render_video(
                outputs['radiance_field'][0], bg_color=video_bg_color)['color']
            imageio.mimsave("sample_rf.mp4", video, fps=30)
            print("Radiance field video generated")
        if render_mesh_video:
            video = render_utils.render_video(
                outputs['mesh'][0], bg_color=video_bg_color)['normal']
            imageio.mimsave("sample_mesh.mp4", video, fps=30)
            print("Mesh video generated")

        if generate_glb:
            # GLB files can be extracted from the outputs
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                # Optional parameters
                simplify=0.95,          # Ratio of triangles to remove in the simplification process
                texture_size=1024,      # Size of the texture used for the GLB
            )

            print("GLB file generated")
            glb.export("sample.glb")

        # Save Gaussians as PLY files
        # outputs['gaussian'][0].save_ply("sample.ply")

        if generate_glb:
            # Read the binary file into bytes
            with open("sample.glb", "rb") as f:
                glb_bytes = f.read()
                glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")
        else:
            glb_base64 = None
            
        # Read the videos if they were generated
        if render_gaussian_video:
            with open("sample_gs.mp4", "rb") as f:
                gs_video_bytes = f.read()
                gs_video_base64 = base64.b64encode(gs_video_bytes).decode("utf-8")
        else:
            gs_video_base64 = None
            
        if render_rf_video:
            with open("sample_rf.mp4", "rb") as f:
                rf_video_bytes = f.read()
                rf_video_base64 = base64.b64encode(rf_video_bytes).decode("utf-8")
        else:
            rf_video_base64 = None
        
        if render_mesh_video:
            with open("sample_mesh.mp4", "rb") as f:
                mesh_video_bytes = f.read()
                mesh_video_base64 = base64.b64encode(mesh_video_bytes).decode("utf-8")
        else:
            mesh_video_base64 = None

        return {
            "glb": glb_base64,
            "gaussian_video": gs_video_base64,
            "radiance_field_video": rf_video_base64,
            "mesh_video": mesh_video_base64,
        }

@app.local_entrypoint()
def main():
    model = modal.Cls.lookup("trellis-3d-generation", "Model")()

    file_path = f"sample.jpeg"
    image = Image.open(file_path)

    result = model.generate_3d.remote(image, seed=42, render_gaussian_video=True,
                                      render_rf_video=True, render_mesh_video=True,
                                      video_bg_color=(1, 1, 1)
                                      )

    # Create a cache directory if it doesn't exist
    os.makedirs(".cache/TRELLIS", exist_ok=True)

    # Write the GLB bytes to a file
    if "glb" in result:
        with open(f".cache/TRELLIS/sample.glb", "wb") as f:
            f.write(base64.b64decode(result["glb"]))
        
    if "gaussian_video" in result:
        with open(f".cache/TRELLIS/sample_gs.mp4", "wb") as f:
            f.write(base64.b64decode(result["gaussian_video"]))
    
    if "radiance_field_video" in result:
        with open(f".cache/TRELLIS/sample_rf.mp4", "wb") as f:
            f.write(base64.b64decode(result["radiance_field_video"]))
    
    if "mesh_video" in result:
        with open(f".cache/TRELLIS/sample_mesh.mp4", "wb") as f:
            f.write(base64.b64decode(result["mesh_video"]))

if __name__ == "__main__":
    print("Running __main__")
    main()