import unittest
from PIL import Image
import numpy as np
import os
from src.deepfakes.facedetect import FaceDetector # Assuming facedetect.py is in src.deepfakes

# Helper to create a dummy image file
def create_dummy_image(path, width=64, height=64):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new('RGB', (width, height), color = 'red')
    img.save(path)

class TestFaceDetector(unittest.TestCase):

    def setUp(self):
        # Create dummy files needed for some tests
        self.dummy_target_image_path = "temp_test_data/target_dummy.jpg"
        self.dummy_source_image_path = "temp_test_data/source_dummy.jpg"
        self.dummy_output_dir = "temp_test_output"

        create_dummy_image(self.dummy_target_image_path)
        # We only create the source dummy if a test specifically needs it to exist.

        # Suppress print statements from FaceDetector during tests if possible,
        # or check for specific print outputs if that's part of the test.
        # For now, we are not capturing stdout.

    def tearDown(self):
        # Clean up dummy files and directories
        if os.path.exists(self.dummy_target_image_path):
            os.remove(self.dummy_target_image_path)
        if os.path.exists(self.dummy_source_image_path):
            os.remove(self.dummy_source_image_path)
        
        # Attempt to remove dirs if they exist and are empty
        if os.path.exists("temp_test_data"):
            try:
                os.rmdir("temp_test_data") # Only removes if empty
            except OSError:
                pass # Directory not empty, or other error
        if os.path.exists(self.dummy_output_dir):
            # Potentially more complex cleanup if output_dir contains files
            # For now, assume it might be created but not necessarily empty
            try:
                # If FaceDetector created files, they would be here.
                # For init tests, it mostly just creates the dir.
                if os.path.isdir(self.dummy_output_dir) and not os.listdir(self.dummy_output_dir):
                    os.rmdir(self.dummy_output_dir)
                elif os.path.isdir(self.dummy_output_dir): # if it has files, more complex
                     pass # print(f"Warning: Output dir {self.dummy_output_dir} not empty, not removing.")
            except OSError:
                pass


    def test_init_no_source(self):
        """Test FaceDetector initialization without a source image."""
        detector = FaceDetector(input_path=self.dummy_target_image_path, output_dir=self.dummy_output_dir)
        self.assertIsNone(detector.source_face_data, "source_face_data should be None when no source_image_path is provided.")
        self.assertIsNone(detector.source_face_np, "source_face_np should be None.")
        self.assertIsNone(detector.source_landmarks_np, "source_landmarks_np should be None.")
        self.assertTrue(os.path.isdir(self.dummy_output_dir), "Output directory should be created.")

    def test_init_source_not_found(self):
        """Test FaceDetector initialization with a non-existent source_image_path."""
        non_existent_source_path = "temp_test_data/non_existent_source.jpg"
        detector = FaceDetector(
            input_path=self.dummy_target_image_path,
            output_dir=self.dummy_output_dir,
            source_image_path=non_existent_source_path
        )
        self.assertIsNone(detector.source_face_data, "source_face_data should be None if source image is not found.")
        self.assertIsNone(detector.source_face_np, "source_face_np should be None.")
        self.assertIsNone(detector.source_landmarks_np, "source_landmarks_np should be None.")
        # We expect a print warning like "[!] Source image not found: ..."
        # Capturing stdout/stderr is more involved, so we're just checking state.

    def test_init_with_valid_source_image_mocked_detection(self):
        """
        Test FaceDetector initialization with a valid source image.
        This test relies on MTCNN not finding faces in a plain red dummy image,
        so source_face_data will be None because no face is detected, not because the file is bad.
        A more advanced test would mock mtcnn.detect.
        """
        create_dummy_image(self.dummy_source_image_path) # Create the dummy source
        detector = FaceDetector(
            input_path=self.dummy_target_image_path,
            output_dir=self.dummy_output_dir,
            source_image_path=self.dummy_source_image_path
        )
        # Since the dummy image is plain red, MTCNN is unlikely to find a face.
        # If it found no face, source_face_data would be None.
        # If it found a face but no landmarks, source_landmarks_np would be None.
        # The key here is that it attempted to load the image.
        # We expect print statements like:
        # "[+] Source face loaded..." (if face found) OR "[!] No faces found in source image..."
        # For a plain red image, it's most likely "[!] No faces found..." or "[!] No landmarks found..."
        
        # Given MTCNN's behavior on a plain image, we expect no face to be detected.
        self.assertIsNone(detector.source_face_data, 
                          "source_face_data should be None if no face is detected in the source image.")
        self.assertIsNone(detector.source_face_np)
        self.assertIsNone(detector.source_landmarks_np)

    # Add more tests for _process_image and _process_video if possible,
    # though they are harder to unit test without significant mocking or actual files
    # with detectable faces and a more controlled environment.

if __name__ == '__main__':
    unittest.main()
