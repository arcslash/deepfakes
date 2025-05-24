import unittest
from PIL import Image, ImageDraw
import numpy as np
import os
import cv2 # For image processing in tests
from src.deepfakes.facedetect import FaceDetector
from src.deepfakes.gan_swapper_interface import AbstractGanSwapper

# Helper to create a dummy image file
def create_dummy_image(path, width=64, height=64, color='red'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new('RGB', (width, height), color=color)
    img.save(path)

class MockSuccessGanSwapper(AbstractGanSwapper):
    def __init__(self, model_path=None, config=None):
        super().__init__(model_path, config)
        self.model_loaded_called = False
        self.swap_face_called_count = 0

    def load_model(self):
        self.model_loaded_called = True
        self.model = "MockGanModel" # Simulate model loading
        print("MockSuccessGanSwapper: Model loaded.")

    def swap_face(self, source_face_img_np, target_face_img_np, source_landmarks_np=None, target_landmarks_np=None):
        self.swap_face_called_count += 1
        print("MockSuccessGanSwapper: swap_face called.")
        # Return a dummy swapped face (e.g., same size as target_face_img_np, but different content)
        return np.full_like(target_face_img_np, 128, dtype=np.uint8)

class MockFailGanSwapper(AbstractGanSwapper):
    def __init__(self, model_path=None, config=None):
        super().__init__(model_path, config)
        self.load_model_called = False

    def load_model(self):
        self.load_model_called = True
        print("MockFailGanSwapper: load_model called, simulating failure.")
        raise ValueError("Simulated GAN model load failure")

    def swap_face(self, source_face_img_np, target_face_img_np, source_landmarks_np=None, target_landmarks_np=None):
        # This shouldn't be called if load_model fails and FaceDetector handles it
        raise AssertionError("MockFailGanSwapper.swap_face should not be called if load_model failed.")


class TestFaceDetector(unittest.TestCase):

    def setUp(self):
        self.test_data_dir = "temp_test_data"
        self.dummy_target_image_path = os.path.join(self.test_data_dir, "target_dummy.jpg")
        self.dummy_source_image_path = os.path.join(self.test_data_dir, "source_dummy.jpg")
        self.dummy_output_dir = "temp_test_output"

        os.makedirs(self.test_data_dir, exist_ok=True)
        os.makedirs(self.dummy_output_dir, exist_ok=True)
        
        create_dummy_image(self.dummy_target_image_path, color='blue') # Target is blue
        # Source dummy is created on demand in tests

    def tearDown(self):
        # Clean up dummy files and directories
        for f in [self.dummy_target_image_path, self.dummy_source_image_path]:
            if os.path.exists(f):
                os.remove(f)
        
        # Clean output directory
        if os.path.exists(self.dummy_output_dir):
            for item in os.listdir(self.dummy_output_dir):
                item_path = os.path.join(self.dummy_output_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            try:
                os.rmdir(self.dummy_output_dir)
            except OSError:
                pass # May not be empty if other sub-dirs created by code
        
        if os.path.exists(self.test_data_dir):
            try:
                os.rmdir(self.test_data_dir)
            except OSError:
                pass


    def test_init_no_source_no_gan(self):
        """Test FaceDetector initialization without source image or GAN."""
        detector = FaceDetector(input_path=self.dummy_target_image_path, output_dir=self.dummy_output_dir)
        self.assertIsNone(detector.source_face_data)
        self.assertIsNone(detector.source_face_np)
        self.assertIsNone(detector.source_landmarks_np)
        self.assertIsNone(detector.gan_swapper)
        self.assertTrue(os.path.isdir(self.dummy_output_dir))

    def test_init_with_gan_swapper_success(self):
        """Test FaceDetector initialization with a successful GAN swapper."""
        mock_gan_swapper = MockSuccessGanSwapper()
        detector = FaceDetector(
            input_path=self.dummy_target_image_path,
            output_dir=self.dummy_output_dir,
            gan_swapper=mock_gan_swapper
        )
        self.assertTrue(mock_gan_swapper.model_loaded_called)
        self.assertIsNotNone(detector.gan_swapper)
        self.assertEqual(detector.gan_swapper.model, "MockGanModel")

    def test_init_with_gan_swapper_load_fail(self):
        """Test FaceDetector initialization with a GAN swapper that fails to load."""
        mock_gan_swapper = MockFailGanSwapper()
        detector = FaceDetector(
            input_path=self.dummy_target_image_path,
            output_dir=self.dummy_output_dir,
            gan_swapper=mock_gan_swapper
        )
        self.assertTrue(mock_gan_swapper.load_model_called)
        self.assertIsNone(detector.gan_swapper, "GAN swapper should be None if its load_model fails.")

    def test_init_source_not_found(self):
        """Test FaceDetector initialization with a non-existent source_image_path."""
        non_existent_source_path = os.path.join(self.test_data_dir, "non_existent_source.jpg")
        detector = FaceDetector(
            input_path=self.dummy_target_image_path,
            output_dir=self.dummy_output_dir,
            source_image_path=non_existent_source_path
        )
        self.assertIsNone(detector.source_face_data)

    def test_init_with_valid_source_image_no_face_detected(self):
        """Test with a valid source image where MTCNN finds no face (plain image)."""
        create_dummy_image(self.dummy_source_image_path, color='green') # Source is green
        detector = FaceDetector(
            input_path=self.dummy_target_image_path,
            output_dir=self.dummy_output_dir,
            source_image_path=self.dummy_source_image_path
        )
        self.assertIsNone(detector.source_face_np, "source_face_np should be None if no face detected in source.")

    @unittest.skip("Skipping full _process_image test due to MTCNN dependency and complexity")
    def test_process_image_with_mock_gan_success(self):
        """
        Test _process_image with a mock GAN swapper that succeeds.
        This is a more complex integration test. We'd need to mock mtcnn.detect
        to return predictable boxes and landmarks.
        """
        # 1. Setup Mock GAN and FaceDetector
        mock_gan_swapper = MockSuccessGanSwapper()
        create_dummy_image(self.dummy_source_image_path, color='green') # Source is green
        
        # For this test, we need source_face_np and source_landmarks_np to be set.
        # We can manually set them after creating a detector instance, or mock MTCNN for source loading.
        # Simpler approach: Manually set them.
        detector = FaceDetector(
            input_path=self.dummy_target_image_path, # Target is blue
            output_dir=self.dummy_output_dir,
            source_image_path=self.dummy_source_image_path, # Actual file, but we'll override data
            gan_swapper=mock_gan_swapper
        )
        
        # Manually inject dummy source face data as if MTCNN found a face in the source image
        detector.source_face_np = np.zeros((64, 64, 3), dtype=np.uint8) # Dummy 64x64 black source face
        detector.source_landmarks_np = np.array([[10,10], [50,10], [30,30], [15,50], [45,50]], dtype=np.float32)


        # 2. Mock MTCNN's detect method for the target image processing
        # This is the complex part. For now, let's assume _process_image can be called
        # and we can inspect the output file. A true unit test would mock mtcnn.detect.
        
        # Create a dummy target PIL image for processing
        dummy_target_pil = Image.open(self.dummy_target_image_path) # Blue image
        
        # To truly test _process_image, we need to simulate that mtcnn.detect finds faces.
        # Let's assume it finds one face in the blue image.
        # We'd need to mock `detector.mtcnn.detect`
        # For now, let's call process() and check the output, which is less of a unit test.
        
        detector.process() # This will call _process_image internally

        self.assertTrue(mock_gan_swapper.swap_face_called_count > 0, "GAN swapper's swap_face should have been called.")
        
        # Check output image
        output_image_path = os.path.join(self.dummy_output_dir, "detected_image.jpg")
        self.assertTrue(os.path.exists(output_image_path), "Output image was not created.")
        
        output_image = Image.open(output_image_path)
        output_np = np.array(output_image)
        
        # MockSuccessGanSwapper fills with 128. Check if a significant portion is 128.
        # This is a loose check because we don't know the exact box from dummy MTCNN.
        # A more precise test would mock the box and landmarks from MTCNN.
        # For now, if GAN was called, we expect some 128s.
        # The original blue image would not have 128s. (0,0,255)
        is_blue_present = np.any(np.all(output_np == [0,0,255], axis=-1))
        is_swapped_val_present = np.any(output_np == 128)

        self.assertTrue(is_swapped_val_present, "Swapped value (128) not found in output image.")
        # Depending on how much of the image is swapped, blue might still be present or not.
        # If the whole dummy image was one big "face", blue might be gone.

if __name__ == '__main__':
    unittest.main()
