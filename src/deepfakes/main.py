from deepfakes.facedetect import FaceDetector


def main():
    # Choose your input file and type
    facedetect = FaceDetector('data/video_test.mp4', input_type='video')
    facedetect.process()


if __name__ == '__main__':
    main()
