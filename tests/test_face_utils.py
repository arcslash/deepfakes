import unittest
import numpy as np
import cv2
from src.deepfakes.faceutils import align_face, create_face_mask

class TestFaceUtils(unittest.TestCase):

    def test_create_mask_valid_landmarks(self):
        """Test create_face_mask with valid landmarks."""
        image_shape = (100, 100, 3)
        # Square landmarks
        landmarks = np.array([
            [20, 20], [80, 20], [80, 80], [20, 80]
        ], dtype=np.float32)

        mask = create_face_mask(image_shape, landmarks)

        self.assertIsNotNone(mask, "Mask should not be None for valid landmarks.")
        self.assertEqual(mask.dtype, np.uint8, "Mask dtype should be uint8.")
        self.assertEqual(mask.shape, (image_shape[0], image_shape[1]), "Mask shape should be (height, width).")
        self.assertTrue(np.sum(mask) > 0, "Mask should contain some white pixels.")
        # Check if the convex hull area is filled (e.g., center pixel)
        # For the square landmarks, the center (50,50) should be white (255)
        self.assertEqual(mask[50, 50], 255, "Center of the mask for a square should be white.")

    def test_create_mask_insufficient_landmarks(self):
        """Test create_face_mask with insufficient landmarks."""
        image_shape = (100, 100, 3)
        # Only 2 landmarks
        landmarks = np.array([
            [20, 20], [80, 20]
        ], dtype=np.float32)

        mask = create_face_mask(image_shape, landmarks)
        self.assertIsNone(mask, "Mask should be None for insufficient landmarks.")

    def test_create_mask_none_landmarks(self):
        """Test create_face_mask with None landmarks."""
        image_shape = (100, 100, 3)
        mask = create_face_mask(image_shape, None)
        self.assertIsNone(mask, "Mask should be None if landmarks are None.")

    def test_align_face_basic(self):
        """Test align_face with basic valid inputs."""
        source_image_np = np.ones((100, 100, 3), dtype=np.uint8) * 255 # White image
        
        source_landmarks_np = np.array([
            [30, 30], [70, 30], [50, 50], [35, 70], [65, 70]
        ], dtype=np.float32)
        
        target_landmarks_np = np.array([
            [35, 35], [75, 35], [55, 55], [40, 75], [70, 75] # Slightly translated
        ], dtype=np.float32)
        
        target_shape = (80, 80) # Different target shape

        warped_face = align_face(source_image_np, source_landmarks_np, target_landmarks_np, target_shape)

        self.assertIsNotNone(warped_face, "Warped face should not be None for valid inputs.")
        self.assertEqual(warped_face.shape[0], target_shape[0], f"Warped face height should be {target_shape[0]}.")
        self.assertEqual(warped_face.shape[1], target_shape[1], f"Warped face width should be {target_shape[1]}.")
        self.assertEqual(warped_face.shape[2], 3, "Warped face should have 3 channels.")

    def test_align_face_insufficient_landmarks(self):
        """Test align_face with insufficient landmarks for source."""
        source_image_np = np.ones((100, 100, 3), dtype=np.uint8) * 255
        source_landmarks_np = np.array([[30, 30], [70, 30]], dtype=np.float32) # 2 points
        target_landmarks_np = np.array([[30, 30], [70, 30], [50, 50]], dtype=np.float32) # 3 points
        target_shape = (100, 100)

        warped_face = align_face(source_image_np, source_landmarks_np, target_landmarks_np, target_shape)
        self.assertIsNone(warped_face, "Warped face should be None if source landmarks are insufficient.")

    def test_align_face_none_landmarks(self):
        """Test align_face with None for landmarks."""
        source_image_np = np.ones((100, 100, 3), dtype=np.uint8) * 255
        valid_landmarks = np.array([[30,30], [70,30], [50,50]], dtype=np.float32)
        target_shape = (100,100)

        warped_face_none_source = align_face(source_image_np, None, valid_landmarks, target_shape)
        self.assertIsNone(warped_face_none_source, "Warped face should be None if source landmarks are None.")

        warped_face_none_target = align_face(source_image_np, valid_landmarks, None, target_shape)
        self.assertIsNone(warped_face_none_target, "Warped face should be None if target landmarks are None.")

if __name__ == '__main__':
    unittest.main()
