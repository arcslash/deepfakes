import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import mmcv
from PIL import Image, ImageDraw
import numpy as np
import os
from .faceutils import align_face, create_face_mask
from .gan_swapper_interface import AbstractGanSwapper # Import AbstractGanSwapper


class FaceDetector:
    def __init__(self,
                 input_path: str,
                 input_type: str = 'image',
                 output_dir: str = 'output',
                 source_image_path: str = None,
                 gan_swapper: AbstractGanSwapper = None): # Added gan_swapper
        self.path = input_path
        self.type = input_type.lower()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device, post_process=True) # Ensure post_process is True for landmarks
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.source_face_data = None
        self.source_face_np = None
        self.source_landmarks_np = None
        self.gan_swapper = gan_swapper # Store gan_swapper

        if self.gan_swapper:
            try:
                print("[+] Loading GAN swapper model...")
                self.gan_swapper.load_model()
                print("[+] GAN swapper model loaded successfully.")
            except Exception as e:
                print(f"[!] Error loading GAN swapper model: {e}. Disabling GAN swapper.")
                self.gan_swapper = None

        if source_image_path:
            try:
                source_image_pil = Image.open(source_image_path).convert("RGB")
                # Detect faces and landmarks
                boxes, _, landmarks_list = self.mtcnn.detect(source_image_pil, landmarks=True)

                if boxes is not None and landmarks_list is not None and len(landmarks_list) > 0:
                    source_box = boxes[0] # Use the first detected face
                    source_landmarks = landmarks_list[0] # Corresponding landmarks

                    # Crop the source face PIL image
                    cropped_pil_image = source_image_pil.crop(source_box)
                    
                    # Convert PIL image to NumPy array (BGR for OpenCV)
                    self.source_face_np = cv2.cvtColor(np.array(cropped_pil_image), cv2.COLOR_RGB2BGR)
                    
                    # Convert landmarks to NumPy array
                    self.source_landmarks_np = np.array(source_landmarks, dtype=np.float32)
                    
                    # Store original PIL cropped image and landmarks if needed elsewhere, or just use the np versions
                    self.source_face_data = {'image': cropped_pil_image, 'landmarks': source_landmarks}

                    print(f'[+] Source face loaded, cropped, converted to NumPy, and landmarks stored from {source_image_path}')
                
                elif boxes is not None: # Faces detected, but no landmarks
                    source_box = boxes[0]
                    cropped_pil_image = source_image_pil.crop(source_box)
                    self.source_face_np = cv2.cvtColor(np.array(cropped_pil_image), cv2.COLOR_RGB2BGR)
                    self.source_face_data = {'image': cropped_pil_image, 'landmarks': None} # Keep PIL image for simple resize if no landmarks
                    self.source_landmarks_np = None
                    print(f'[!] Source face loaded and cropped from {source_image_path}, but NO LANDMARKS detected. Alignment will be skipped.')
                else: # No faces found
                    print(f'[!] No faces found in source image: {source_image_path}')
            except FileNotFoundError:
                print(f'[!] Source image not found: {source_image_path}')
            except Exception as e:
                print(f'[!] Failed to load source image: {e}')

    def process(self):
        if self.type == 'image':
            self._process_image()
        elif self.type == 'video':
            self._process_video()
        else:
            print(f'[!] Error: Invalid input type "{self.type}". Must be "image" or "video".')

    def _process_image(self):
        print('[+] Processing Image')
        try:
            img = Image.open(self.path).convert("RGB")
        except Exception as e:
            print(f'[!] Failed to load image: {e}')
            return

        boxes, _, landmarks_list = self.mtcnn.detect(img, landmarks=True) # Ensure landmarks=True
        if boxes is None:
            print('[!] No faces detected.')
            return

        # Process with OpenCV from the start
        frame_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        if landmarks_list is not None:
            for i, (box, target_landmarks_points) in enumerate(zip(boxes, landmarks_list)):
                target_landmarks_np_abs = np.array(target_landmarks_points, dtype=np.float32) # Absolute landmarks
                
                # Define target box coordinates
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                target_box_width = x2 - x1
                target_box_height = y2 - y1

                if self.gan_swapper and self.source_face_np is not None and target_box_width > 0 and target_box_height > 0:
                    print(f"[+] Attempting GAN swap for face {i}")
                    target_face_crop_np = frame_np[y1:y2, x1:x2]
                    
                    try:
                        swapped_face_np = self.gan_swapper.swap_face(
                            self.source_face_np,
                            target_face_crop_np,
                            self.source_landmarks_np, # Pass source landmarks
                            target_landmarks_np_abs  # Pass absolute target landmarks
                        )

                        if swapped_face_np is not None and \
                           swapped_face_np.shape[0] > 0 and swapped_face_np.shape[1] > 0:
                            # Resize GAN output to fit the target box if necessary
                            if swapped_face_np.shape[0] != target_box_height or swapped_face_np.shape[1] != target_box_width:
                                print(f"[+] Resizing GAN output from {swapped_face_np.shape[:2]} to {(target_box_height, target_box_width)}")
                                swapped_face_np = cv2.resize(swapped_face_np, (target_box_width, target_box_height), interpolation=cv2.INTER_AREA)
                            
                            frame_np[y1:y2, x1:x2] = swapped_face_np
                            print(f"[+] GAN swap successful for face {i}")
                        else:
                            print(f"[!] GAN swap returned None or invalid image for face {i}. Skipping.")
                            # Optionally, draw a different color rectangle for GAN failure before fallback
                            cv2.rectangle(frame_np, (x1, y1), (x2, y2), (255, 0, 255), 2) # Magenta for GAN failure

                    except Exception as e:
                        print(f"[!] Error during GAN swap for face {i}: {e}. Falling back to geometric.")
                        # Fallback to geometric if GAN fails
                        self._apply_geometric_swap(frame_np, box, target_landmarks_np_abs, target_box_height, target_box_width, i)

                elif self.source_face_np is not None and self.source_landmarks_np is not None and target_box_width > 0 and target_box_height > 0:
                    # Fallback to geometric if no GAN swapper or no source face for GAN
                    self._apply_geometric_swap(frame_np, box, target_landmarks_np_abs, target_box_height, target_box_width, i)
                
                elif self.source_face_data and self.source_face_data.get('image') is not None: # Fallback to simple PIL resize
                    print("[!] Source landmarks not available for geometric swap. Falling back to simple PIL resize and paste.")
                    # This part needs to be done on a PIL version of the frame if we are using PIL methods
                    pil_temp_frame = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
                    target_size = (target_box_width, target_box_height)
                    resized_source_face_pil = self.source_face_data['image'].resize(target_size, Image.ANTIALIAS)
                    pil_temp_frame.paste(resized_source_face_pil, (x1, y1))
                    frame_np = cv2.cvtColor(np.array(pil_temp_frame), cv2.COLOR_RGB2BGR)
                else: # No source face at all
                    cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 0, 255), 6) # Red for no source
                
                # Draw landmarks (on a PIL version for convenience if _draw_landmarks expects PIL)
                pil_for_landmarks = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
                draw_pil = ImageDraw.Draw(pil_for_landmarks)
                self._draw_landmarks(draw_pil, target_landmarks_points) 
                frame_np = cv2.cvtColor(np.array(pil_for_landmarks), cv2.COLOR_RGB2BGR)


        elif boxes is not None: # No landmarks detected in target image, but boxes are present
            for box in boxes:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if self.source_face_data and self.source_face_data.get('image') is not None: # Fallback to simple PIL resize
                     pil_temp_frame = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
                     target_size = (x2-x1, y2-y1)
                     resized_source_face_pil = self.source_face_data['image'].resize(target_size, Image.ANTIALIAS)
                     pil_temp_frame.paste(resized_source_face_pil, (x1, y1))
                     frame_np = cv2.cvtColor(np.array(pil_temp_frame), cv2.COLOR_RGB2BGR)
                else: # No source face
                    cv2.rectangle(frame_np, (x1,y1), (x2,y2), (0,0,255), 6)
                print("[!] Landmarks not detected for a face in _process_image (target). Drawing box.")

        # Convert final frame_np (BGR) back to PIL Image (RGB) for saving
        final_pil_image = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
        output_path = os.path.join(self.output_dir, 'detected_image.jpg')
        final_pil_image.save(output_path)
        print(f'[✓] Saved output to {output_path}')

    def _apply_geometric_swap(self, frame_np, box, target_landmarks_np_abs, target_box_height, target_box_width, face_idx):
        """Helper function to apply the original geometric swap."""
        print(f"[_apply_geometric_swap] Applying geometric swap for face {face_idx}")
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        warped_source_face_np = align_face(
            self.source_face_np,
            self.source_landmarks_np,
            target_landmarks_np_abs,
            target_shape=(target_box_height, target_box_width)
        )

        if warped_source_face_np is not None:
            target_landmarks_np_relative = target_landmarks_np_abs - np.array([x1, y1], dtype=np.float32)
            face_mask_np = create_face_mask(warped_source_face_np.shape, target_landmarks_np_relative)

            if face_mask_np is not None:
                if len(warped_source_face_np.shape) == 3 and len(face_mask_np.shape) == 2:
                    face_mask_np = cv2.cvtColor(face_mask_np, cv2.COLOR_GRAY2BGR)
                face_mask_np = face_mask_np.astype(np.uint8)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                if len(face_mask_np.shape) == 3:
                    face_mask_for_blend_np = cv2.cvtColor(face_mask_np, cv2.COLOR_BGR2GRAY)
                else:
                    face_mask_for_blend_np = face_mask_np
                face_mask_for_blend_np = np.where(face_mask_for_blend_np > 0, 255, 0).astype(np.uint8)
                
                h_mask, w_mask = face_mask_for_blend_np.shape
                h_frame, w_frame = frame_np.shape[:2]
                clone_x_start = center[0] - w_mask // 2; clone_y_start = center[1] - h_mask // 2
                clone_x_end = clone_x_start + w_mask; clone_y_end = clone_y_start + h_mask

                if clone_x_start >= 0 and clone_y_start >=0 and clone_x_end <= w_frame and clone_y_end <= h_frame and \
                   warped_source_face_np.shape[0] == h_mask and warped_source_face_np.shape[1] == w_mask:
                    try:
                        frame_np[y1:y2, x1:x2] = cv2.seamlessClone( # Modify frame_np directly in the ROI
                            warped_source_face_np, frame_np[y1:y2, x1:x2], face_mask_for_blend_np, 
                            (target_box_width // 2, target_box_height // 2), # Center of ROI
                            cv2.NORMAL_CLONE
                        )
                    except cv2.error as e:
                        print(f"[!] Error during seamlessClone in geometric fallback: {e}. Pasting directly.")
                        masked_warped_source = cv2.bitwise_and(warped_source_face_np, warped_source_face_np, mask=face_mask_for_blend_np)
                        target_roi_geo = frame_np[y1:y2, x1:x2]
                        if len(target_roi_geo.shape) == 3 and len(face_mask_for_blend_np.shape) == 2:
                             face_mask_for_blend_np_3ch_geo = cv2.cvtColor(face_mask_for_blend_np, cv2.COLOR_GRAY2BGR)
                        else:
                             face_mask_for_blend_np_3ch_geo = face_mask_for_blend_np
                        inverse_mask_roi_geo = cv2.bitwise_not(face_mask_for_blend_np_3ch_geo)
                        masked_target_roi_geo = cv2.bitwise_and(target_roi_geo, inverse_mask_roi_geo)
                        combined_roi_geo = cv2.add(masked_warped_source, masked_target_roi_geo)
                        frame_np[y1:y2, x1:x2] = combined_roi_geo
                else: # ROI for seamless clone out of bounds
                    print("[!] ROI for seamlessClone (geometric) is outside frame or size mismatch. Pasting directly with mask.")
                    masked_warped_source = cv2.bitwise_and(warped_source_face_np, warped_source_face_np, mask=face_mask_for_blend_np)
                    target_roi_geo = frame_np[y1:y2, x1:x2]
                    if len(target_roi_geo.shape) == 3 and len(face_mask_for_blend_np.shape) == 2:
                        face_mask_for_blend_np_3ch_geo = cv2.cvtColor(face_mask_for_blend_np, cv2.COLOR_GRAY2BGR)
                    else:
                        face_mask_for_blend_np_3ch_geo = face_mask_for_blend_np
                    inverse_mask_roi_geo = cv2.bitwise_not(face_mask_for_blend_np_3ch_geo)
                    masked_target_roi_geo = cv2.bitwise_and(target_roi_geo, inverse_mask_roi_geo)
                    combined_roi_geo = cv2.add(masked_warped_source, masked_target_roi_geo)
                    frame_np[y1:y2, x1:x2] = combined_roi_geo
            else: # Mask creation failed in geometric
                print("[!] Mask creation failed (geometric). Pasting aligned face directly.")
                if warped_source_face_np.shape[0] == target_box_height and warped_source_face_np.shape[1] == target_box_width:
                     frame_np[y1:y2, x1:x2] = warped_source_face_np
                else:
                     print("[!] Size mismatch in direct paste (geometric) after mask creation failure. Skipping paste.")
        else: # Alignment failed in geometric
            print(f"[!] Alignment failed (geometric) for face {face_idx}, drawing green rectangle.")
            cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 255, 0), 3)


    def _process_video(self):
        print('[+] Processing Video')
        try:
            video = mmcv.VideoReader(self.path)
        except Exception as e:
            print(f'[!] Failed to load video: {e}')
            return

        frames_tracked = []
        for frame_idx, frame in enumerate(video): # Use frame_idx for clarity
            print(f'\r[→] Tracking frame {frame_idx + 1}/{len(video)}', end='')
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_frame_np = frame.copy() 

            boxes, _, landmarks_list = self.mtcnn.detect(pil_frame, landmarks=True)
            
            if landmarks_list is not None:
                for face_idx_in_frame, (box, target_landmarks_points) in enumerate(zip(boxes, landmarks_list)):
                    target_landmarks_np_abs = np.array(target_landmarks_points, dtype=np.float32)
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    target_box_width = x2 - x1
                    target_box_height = y2 - y1

                    if self.gan_swapper and self.source_face_np is not None and target_box_width > 0 and target_box_height > 0:
                        print(f"[+] Attempting GAN swap for frame {frame_idx}, face {face_idx_in_frame}")
                        target_face_crop_np = current_frame_np[y1:y2, x1:x2]
                        try:
                            swapped_face_np = self.gan_swapper.swap_face(
                                self.source_face_np,
                                target_face_crop_np,
                                self.source_landmarks_np,
                                target_landmarks_np_abs
                            )
                            if swapped_face_np is not None and swapped_face_np.shape[0] > 0 and swapped_face_np.shape[1] > 0:
                                if swapped_face_np.shape[0] != target_box_height or swapped_face_np.shape[1] != target_box_width:
                                    swapped_face_np = cv2.resize(swapped_face_np, (target_box_width, target_box_height), interpolation=cv2.INTER_AREA)
                                current_frame_np[y1:y2, x1:x2] = swapped_face_np
                            else:
                                cv2.rectangle(current_frame_np, (x1, y1), (x2, y2), (255, 0, 255), 2) # Magenta
                        except Exception as e:
                            print(f"[!] Error during GAN swap for frame {frame_idx}, face {face_idx_in_frame}: {e}. Falling back.")
                            self._apply_geometric_swap(current_frame_np, box, target_landmarks_np_abs, target_box_height, target_box_width, face_idx_in_frame)
                    
                    elif self.source_face_np is not None and self.source_landmarks_np is not None and target_box_width > 0 and target_box_height > 0:
                        self._apply_geometric_swap(current_frame_np, box, target_landmarks_np_abs, target_box_height, target_box_width, face_idx_in_frame)
                    
                    elif self.source_face_data and self.source_face_data.get('image') is not None: # Fallback to PIL resize
                        pil_temp_frame = Image.fromarray(cv2.cvtColor(current_frame_np, cv2.COLOR_BGR2RGB))
                        target_size = (target_box_width, target_box_height)
                        resized_source_face_pil = self.source_face_data['image'].resize(target_size, Image.ANTIALIAS)
                        pil_temp_frame.paste(resized_source_face_pil, (x1, y1))
                        current_frame_np = cv2.cvtColor(np.array(pil_temp_frame), cv2.COLOR_RGB2BGR)
                    else: # No source face
                        cv2.rectangle(current_frame_np, (x1, y1), (x2, y2), (0, 0, 255), 6)
                    
                    # Draw landmarks (on a PIL version for convenience)
                    pil_for_landmarks = Image.fromarray(cv2.cvtColor(current_frame_np, cv2.COLOR_BGR2RGB))
                    draw_pil = ImageDraw.Draw(pil_for_landmarks)
                    self._draw_landmarks(draw_pil, target_landmarks_points)
                    current_frame_np = cv2.cvtColor(np.array(pil_for_landmarks), cv2.COLOR_RGB2BGR)

            elif boxes is not None: # No landmarks in target frame, but boxes present
                 for face_idx_in_frame, box in enumerate(boxes): # Use different index
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    if self.source_face_data and self.source_face_data.get('image') is not None: # Fallback to PIL resize
                        pil_temp_frame = Image.fromarray(cv2.cvtColor(current_frame_np, cv2.COLOR_BGR2RGB))
                        target_size = (x2-x1, y2-y1)
                        resized_source_face_pil = self.source_face_data['image'].resize(target_size, Image.ANTIALIAS)
                        pil_temp_frame.paste(resized_source_face_pil, (x1, y1))
                        current_frame_np = cv2.cvtColor(np.array(pil_temp_frame), cv2.COLOR_RGB2BGR)
                    else:
                        cv2.rectangle(current_frame_np, (x1,y1), (x2,y2), (0,0,255), 6)
                    print(f"[!] Landmarks not detected for a face in frame {frame_idx}, face {face_idx_in_frame} of _process_video (target).")

            # Convert final current_frame_np (BGR) to PIL Image (RGB) for resizing and video writing
            final_pil_frame = Image.fromarray(cv2.cvtColor(current_frame_np, cv2.COLOR_BGR2RGB))
            resized_pil_frame = final_pil_frame.resize((640, 360), Image.BILINEAR)
            frames_tracked.append(resized_pil_frame)

        print('\n[✓] Finished face tracking. Writing video...')

        output_path = os.path.join(self.output_dir, 'detected_video.mp4')
        dim = frames_tracked[0].size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, 25.0, dim)

        for frame in frames_tracked:
            out_video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        out_video.release()
        print(f'[✓] Saved video to {output_path}')

    def _draw_landmarks(self, draw_context, landmarks_set):
        """
        Helper function to draw landmarks on an image.
        :param draw_context: PIL ImageDraw object.
        :param landmarks_set: A set of (x, y) tuples for landmarks.
        """
        if landmarks_set is None:
            return
        for landmark_point in landmarks_set:
            # Define the bounding box for the ellipse
            x, y = landmark_point
            radius = 2 # Radius of the circle
            # Define the bounding box for the ellipse (circle)
            # For some reason, the ImageDraw.ellipse method takes a bounding box
            # (x0, y0, x1, y1) instead of a center and radius.
            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw_context.ellipse(bbox, fill='yellow', outline='yellow')
