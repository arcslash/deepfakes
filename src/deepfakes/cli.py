import typer
from deepfakes.facedetect import FaceDetector

app = typer.Typer(help="Deepfake Face Detection CLI")


@app.command()
def detect(
        path: str = typer.Argument(..., help="Path to image or video file."),
        input_type: str = typer.Option("image", "--type", "-t", help="Input type: image or video."),
        output_dir: str = typer.Option("output", "--output", "-o", help="Output directory.")
):
    """
    Run face detection on the given image or video file.
    """
    typer.echo(f"[+] Starting face detection on: {path} ({input_type})")
    detector = FaceDetector(path, input_type=input_type, output_dir=output_dir)
    detector.process()


if __name__ == "__main__":
    app()
