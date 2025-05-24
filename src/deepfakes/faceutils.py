import cv2
import numpy as np

def align_face(source_image_np: np.ndarray, source_landmarks_np: np.ndarray, target_landmarks_np: np.ndarray, target_shape: tuple) -> np.ndarray | None:
    """
    Aligns a source face to target landmarks using an affine transformation.

    Args:
        source_image_np: The source face image as a NumPy array (OpenCV format, BGR).
        source_landmarks_np: NumPy array of landmarks for the source face.
        target_landmarks_np: NumPy array of landmarks for the target face.
        target_shape: Tuple (height, width) for the output warped image.

    Returns:
        The warped source face as a NumPy array (OpenCV format, BGR), or None if alignment fails.
    """
    if source_landmarks_np is None or target_landmarks_np is None:
        print("[!] Align_face: Source or target landmarks are None.")
        return None
    if source_landmarks_np.shape[0] < 3 or target_landmarks_np.shape[0] < 3:
        print("[!] Align_face: Not enough landmarks for transformation (need at least 3).")
        return None

    try:
        # Ensure landmarks are float32
        source_landmarks_np = source_landmarks_np.astype(np.float32)
        target_landmarks_np = target_landmarks_np.astype(np.float32)

        # Estimate affine transformation.
        # Using estimateAffinePartial2D as it's more robust to outliers and doesn't require exactly 3 points.
        # It returns a tuple (transformation_matrix, inliers_mask)
        transformation_matrix, inliers = cv2.estimateAffinePartial2D(source_landmarks_np, target_landmarks_np, method=cv2.LMEDS)

        if transformation_matrix is None:
            print("[!] Align_face: estimateAffinePartial2D failed to compute transformation matrix.")
            return None

        # Get the dimensions for the output warped image
        h, w = target_shape

        # Apply the affine transformation to warp the source image.
        # The output size (dsize) should be the size of the area where the target face is,
        # or the size of the target image if we want to place the warped face onto it directly.
        # For now, using the shape of the original source image.
        # This might need adjustment depending on how it's pasted.
        warped_source_face_np = cv2.warpAffine(source_image_np, transformation_matrix, (w, h))

        return warped_source_face_np

    except Exception as e:
        print(f"[!] Align_face: Error during affine transformation: {e}")
        return None

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    # Create dummy source image (e.g., 100x100, 3 channels)
    dummy_source_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_source_img[20:80, 20:80] = [255, 0, 0] # Blue square

    # Dummy landmarks (e.g., 5 points for eyes, nose, mouth corners)
    # These would typically come from a landmark detector
    source_pts = np.array([
        [30, 30], [70, 30], [50, 50], [35, 70], [65, 70]
    ], dtype=np.float32)

    # Dummy target landmarks (slightly shifted and scaled)
    target_pts = np.array([
        [40, 40], [80, 40], [60, 60], [45, 80], [75, 80]
    ], dtype=np.float32)

    print("Testing align_face function...")
    warped_face = align_face(dummy_source_img, source_pts, target_pts, target_shape=(dummy_source_img.shape[0], dummy_source_img.shape[1]))

    if warped_face is not None:
        print("Alignment successful. Displaying warped face.")
        # In a real scenario, you'd save or use this image.
        # For testing, let's try to save it.
        cv2.imwrite("warped_test_face.jpg", warped_face)
        print("Saved test warped image to warped_test_face.jpg")

        # To display, you'd typically use:
        # cv2.imshow("Original Source", dummy_source_img)
        # cv2.imshow("Warped Face", warped_face)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("Alignment failed.")

    # Test with insufficient landmarks
    source_pts_insufficient = np.array([[30,30], [70,30]], dtype=np.float32)
    warped_face_insufficient = align_face(dummy_source_img, source_pts_insufficient, target_pts, target_shape=(100,100))
    if warped_face_insufficient is None:
        print("Alignment correctly failed for insufficient landmarks.")

    # Test with None landmarks
    warped_face_none = align_face(dummy_source_img, None, target_pts, target_shape=(100,100))
    if warped_face_none is None:
        print("Alignment correctly failed for None landmarks.")

    print("Test complete. Check for warped_test_face.jpg if alignment was successful.")


def create_face_mask(image_shape: tuple, landmarks_np: np.ndarray) -> np.ndarray | None:
    """
    Creates a binary mask from facial landmarks using a convex hull.

    Args:
        image_shape: Tuple (height, width, channels) or (height, width) of the image 
                     for which the mask is being created.
        landmarks_np: NumPy array of facial landmarks, assumed to be relative to the image_shape.

    Returns:
        A binary mask (NumPy array, uint8, single channel) with the convex hull filled in white (255),
        or None if mask creation fails.
    """
    if landmarks_np is None:
        print("[!] Create_face_mask: Landmarks are None.")
        return None
    if landmarks_np.shape[0] < 3: # Convex hull needs at least 3 points
        print("[!] Create_face_mask: Not enough landmarks to form a convex hull (need at least 3).")
        return None

    try:
        # Create a black image (mask)
        if len(image_shape) == 3:
            h, w, _ = image_shape
        elif len(image_shape) == 2:
            h, w = image_shape
        else:
            print("[!] Create_face_mask: Invalid image_shape format.")
            return None
            
        mask = np.zeros((h, w), dtype=np.uint8)

        # Compute convex hull of the landmarks.
        # Landmarks should be integers for fillConvexPoly.
        hull = cv2.convexHull(landmarks_np.astype(np.int32))

        # Draw the filled convex hull in white on the mask.
        cv2.fillConvexPoly(mask, hull, 255)

        return mask

    except Exception as e:
        print(f"[!] Create_face_mask: Error during mask creation: {e}")
        return None

if __name__ == '__main__':
    # ... (keep existing align_face test code) ...

    # Test create_face_mask
    print("\nTesting create_face_mask function...")
    dummy_mask_shape = (100, 100, 3)
    # These landmarks are now relative to the 100x100 image
    relative_landmarks = np.array([
        [10, 10], [90, 10], [50, 30], [20, 80], [80, 80] 
    ], dtype=np.float32)

    mask = create_face_mask(dummy_mask_shape, relative_landmarks)
    if mask is not None:
        print("Mask creation successful.")
        cv2.imwrite("convex_hull_mask.jpg", mask)
        print("Saved test mask to convex_hull_mask.jpg")
        # cv2.imshow("Test Mask", mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("Mask creation failed.")

    mask_insufficient = create_face_mask(dummy_mask_shape, relative_landmarks[:2])
    if mask_insufficient is None:
        print("Mask creation correctly failed for insufficient landmarks.")
    
    mask_none = create_face_mask(dummy_mask_shape, None)
    if mask_none is None:
        print("Mask creation correctly failed for None landmarks.")
    
    print("Faceutils test complete. Check for generated images.")

        # To display, you'd typically use:
    # Existing align_face test code ...
    if warped_face is not None:
        # print("Alignment successful. Displaying warped face.") # Commented out for brevity
        cv2.imwrite("warped_test_face.jpg", warped_face)
        # print("Saved test warped image to warped_test_face.jpg")
    # else:
        # print("Alignment failed.")

    # Test with insufficient landmarks
    # ...
    # Test with None landmarks
    # ...
    # print("Test complete. Check for warped_test_face.jpg if alignment was successful.") # Commented out
