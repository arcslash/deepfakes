import typer
from deepfakes.facedetect import FaceDetector
from deepfakes.simswap_wrapper import SimSwapWrapper # Import SimSwapWrapper
# from deepfakes.gan_swapper_interface import AbstractGanSwapper # For type hinting if needed

app = typer.Typer(help="Deepfake Face Detection and Swapping CLI")


@app.command()
def detect(
        path: str = typer.Argument(..., help="Path to target image or video file."),
        input_type: str = typer.Option("image", "--type", "-t", help="Input type: image or video."),
        output_dir: str = typer.Option("output", "--output", "-o", help="Output directory."),
        source_image_path: str = typer.Option(None, "--source", "-s", help="Path to the source image for face swapping."),
        gan_model_name: str = typer.Option(None, "--gan-model", help="Specify GAN model (e.g., 'simswap')."),
        gan_model_path: str = typer.Option(None, "--gan-weights", help="Path to GAN model weights (e.g., .pth file)."),
        gan_config_path: str = typer.Option(None, "--gan-config", help="Path to GAN-specific config file (e.g., for SimSwap options or ArcFace model path)."),
        simswap_root_path: str = typer.Option(None, "--simswap-root", help="Path to the root of cloned SimSwap repository (if using SimSwap and needed for its imports)."),
        arcface_model_path: str = typer.Option(None, "--arcface-path", help="Path to ArcFace model weights (e.g. for SimSwap).") # Added for ArcFace
):
    """
    Run face detection on the given image or video file.
    Optionally, provide a source image and/or GAN model for face swapping.
    """
    
    gan_swapper = None
    if gan_model_name:
        if gan_model_name.lower() == "simswap":
            if not gan_model_path:
                typer.echo("[!] Error: --gan-weights is required for SimSwap.", err=True)
                raise typer.Exit(code=1)
            
            typer.echo(f"[+] Initializing SimSwap GAN swapper...")
            simswap_config = {}
            if gan_config_path: # This could be a SimSwap specific options YAML/JSON
                simswap_config['simswap_config_path'] = gan_config_path 
            if arcface_model_path: # Specific config for ArcFace model path
                simswap_config['arcface_model_path'] = arcface_model_path
            else:
                typer.echo("[!] Warning: --arcface-path not provided for SimSwap. The wrapper might use a default or fail if it's required.", err=True)

            try:
                gan_swapper = SimSwapWrapper(
                    model_path=gan_model_path,
                    config=simswap_config,
                    simswap_root_path=simswap_root_path
                )
                typer.echo("[+] SimSwapWrapper initialized.")
            except Exception as e:
                typer.echo(f"[!] Error initializing SimSwapWrapper: {e}", err=True)
                raise typer.Exit(code=1)
        else:
            typer.echo(f"[!] Warning: Unknown GAN model '{gan_model_name}'. Proceeding without GAN.", err=True)

    if source_image_path:
        if gan_swapper:
            typer.echo(f"[+] Starting GAN-based face swapping on: {path} ({input_type}) with source: {source_image_path} using {gan_model_name}")
        else:
            typer.echo(f"[+] Starting geometric face swapping on: {path} ({input_type}) with source: {source_image_path}")
    else:
        typer.echo(f"[+] Starting face detection only on: {path} ({input_type})")
    
    detector = FaceDetector(
        input_path=path,
        input_type=input_type,
        output_dir=output_dir,
        source_image_path=source_image_path,
        gan_swapper=gan_swapper # Pass the initialized swapper
    )
    detector.process()


if __name__ == "__main__":
    app()
