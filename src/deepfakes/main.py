from deepfakes.facedetect import FaceDetector


def main():
    # Choose your input file and type
    # Replace 'data/source_face.jpg' with the actual path to your source image for swapping.
    # Ensure both the input video/image and the source image exist at the specified paths.
    facedetect = FaceDetector(
        input_path='data/video_test.mp4',
        input_type='video',
        output_dir='output',  # Example output directory
        source_image_path='data/source_face.jpg'  # Replace with your source image
    )
    facedetect.process()


if __name__ == '__main__':
    main()
