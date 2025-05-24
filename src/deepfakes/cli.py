import typer
from deepfakes.facedetect import FaceDetector

app = typer.Typer(help="Deepfake Face Detection CLI")


@app.command()
def detect(
        path: str = typer.Argument(..., help="Path to image or video file."),
        input_type: str = typer.Option("image", "--type", "-t", help="Input type: image or video."),
        output_dir: str = typer.Option("output", "--output", "-o", help="Output directory."),
        source_image_path: str = typer.Option(None, "--source", "-s", help="Path to the source image for face swapping.")
):
    """
    Run face detection on the given image or video file.
    Optionally, provide a source image for face swapping.
    """
    if source_image_path:
        typer.echo(f"[+] Starting face detection/swapping on: {path} ({input_type}) with source: {source_image_path}")
    else:
        typer.echo(f"[+] Starting face detection on: {path} ({input_type})")
    
    detector = FaceDetector(
        input_path=path,
        input_type=input_type,
        output_dir=output_dir,
        source_image_path=source_image_path
    )
    detector.process()


if __name__ == "__main__":
    app()
