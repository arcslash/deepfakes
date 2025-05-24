from deepfakes.facedetect import FaceDetector
from deepfakes.simswap_wrapper import SimSwapWrapper # Import SimSwapWrapper

def main():
    # --- Configuration ---
    target_input_path = 'data/video_test.mp4' # Replace with your target image/video
    target_input_type = 'video'               # 'image' or 'video'
    source_face_image_path = 'data/source_face.jpg' # Replace with your source face image

    # Output directory
    output_directory = 'output'

    # --- Optional: GAN Swapper Configuration (SimSwap Example) ---
    # To use GAN-based swapping, uncomment and configure the following:

    # 1. Set to True to enable SimSwap, False to use geometric swap or no swap if source_face_image_path is None
    use_simswap_gan = False # CHANGE THIS TO True TO TRY SIMSWAP

    # 2. Provide the correct paths for your SimSwap setup
    simswap_model_weights_path = "path/to/your/simswap_checkpoint.pth" # IMPORTANT: Replace with actual path
    simswap_repository_root_path = "/path/to/your/cloned/SimSwap_repo"  # IMPORTANT: Replace (if SimSwap needs its modules)
    arcface_model_weights_path = "path/to/your/arcface_model.tar"      # IMPORTANT: Replace (SimSwap uses ArcFace)

    swapper_instance = None
    if use_simswap_gan:
        if not all([simswap_model_weights_path.startswith("path/to/your"), 
                    arcface_model_weights_path.startswith("path/to/your")]):
            
            typer_echo_like_message = print # Using print as typer is not imported here
            
            typer_echo_like_message("[+] Initializing SimSwapWrapper with provided paths...")
            simswap_wrapper_config = {
                'arcface_model_path': arcface_model_weights_path
                # Add any other SimSwap specific config keys here if your wrapper needs them
                # e.g., 'simswap_options_path': 'path/to/simswap_options.yaml'
            }
            try:
                swapper_instance = SimSwapWrapper(
                    model_path=simswap_model_weights_path,
                    config=simswap_wrapper_config,
                    simswap_root_path=simswap_repository_root_path
                )
                typer_echo_like_message("[+] SimSwapWrapper instance created.")
            except Exception as e:
                typer_echo_like_message(f"[!] Error creating SimSwapWrapper: {e}. Check paths and SimSwap setup.")
                swapper_instance = None # Ensure it's None if init fails
        else:
            print("[!] SimSwap paths are placeholders. Please update them to use SimSwap.")
            print("    Geometric swapping will be used if a source image is provided.")


    # --- Initialize and Run FaceDetector ---
    print(f"[+] Starting processing for: {target_input_path}")
    if source_face_image_path:
        if swapper_instance:
            print(f"[+] Using SimSwap for face swapping with source: {source_face_image_path}")
        else:
            print(f"[+] Using geometric face swapping with source: {source_face_image_path}")
    
    facedetect = FaceDetector(
        input_path=target_input_path,
        input_type=target_input_type,
        output_dir=output_directory,
        source_image_path=source_face_image_path,
        gan_swapper=swapper_instance # Pass the GAN swapper instance
    )
    facedetect.process()
    print(f"[✓] Processing complete. Output saved to '{output_directory}'.")


if __name__ == '__main__':
    main()
