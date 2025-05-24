import unittest
import numpy as np
from src.deepfakes.gan_swapper_interface import AbstractGanSwapper

# Extracted DummyGanSwapper for testing purposes
class DummyGanSwapper(AbstractGanSwapper):
    def __init__(self, model_path=None, config=None):
        super().__init__(model_path, config)
        self.load_model_called = False
        self.swap_face_called = False

    def load_model(self):
        # print(f"Dummy model loaded from {self.model_path if self.model_path else 'default path'}")
        self.model = "DummyModelInstance"
        self.load_model_called = True

    def swap_face(self,
                  source_face_img_np: np.ndarray,
                  target_face_img_np: np.ndarray,
                  source_landmarks_np: np.ndarray = None,
                  target_landmarks_np: np.ndarray = None) -> np.ndarray:
        # print("Dummy swap_face called.")
        # print(f"Source shape: {source_face_img_np.shape}, Target shape: {target_face_img_np.shape}")
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        self.swap_face_called = True
        # In a real implementation, GAN processing would happen here.
        # For this dummy, just return the target face modified to indicate swap.
        return np.full_like(target_face_img_np, 123, dtype=np.uint8)


class TestDummyGanSwapper(unittest.TestCase):

    def test_dummy_swapper_initialization(self):
        """Test initialization of DummyGanSwapper."""
        swapper = DummyGanSwapper(model_path="dummy/path", config={"key": "value"})
        self.assertEqual(swapper.model_path, "dummy/path")
        self.assertEqual(swapper.config, {"key": "value"})
        self.assertIsNone(swapper.model)
        self.assertFalse(swapper.load_model_called)
        self.assertFalse(swapper.swap_face_called)

    def test_dummy_swapper_load_model(self):
        """Test the load_model method of DummyGanSwapper."""
        swapper = DummyGanSwapper()
        swapper.load_model()
        self.assertTrue(swapper.load_model_called)
        self.assertEqual(swapper.model, "DummyModelInstance")

    def test_dummy_swapper_swap_face(self):
        """Test the swap_face method of DummyGanSwapper."""
        swapper = DummyGanSwapper()
        swapper.load_model() # Model must be loaded first

        dummy_source_face = np.zeros((128, 128, 3), dtype=np.uint8)
        dummy_target_face = np.ones((128, 128, 3), dtype=np.uint8) * 255

        result_face = swapper.swap_face(dummy_source_face, dummy_target_face)

        self.assertTrue(swapper.swap_face_called)
        self.assertIsNotNone(result_face)
        self.assertEqual(result_face.shape, dummy_target_face.shape)
        # Check if the dummy modification was applied
        self.assertTrue(np.all(result_face == 123))

    def test_dummy_swapper_swap_face_without_load(self):
        """Test that swap_face raises ValueError if model is not loaded."""
        swapper = DummyGanSwapper()
        dummy_source_face = np.zeros((128, 128, 3), dtype=np.uint8)
        dummy_target_face = np.ones((128, 128, 3), dtype=np.uint8) * 255

        with self.assertRaises(ValueError) as context:
            swapper.swap_face(dummy_source_face, dummy_target_face)
        
        self.assertTrue("Model not loaded" in str(context.exception))

if __name__ == '__main__':
    unittest.main()
