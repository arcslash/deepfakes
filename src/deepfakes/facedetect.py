import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import mmcv
from PIL import Image, ImageDraw
import numpy as np
import os


class FaceDetector:
    def __init__(self, input_path: str, input_type: str = 'image', output_dir: str = 'output'):
        self.path = input_path
        self.type = input_type.lower()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

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

        boxes, _ = self.mtcnn.detect(img)
        if boxes is None:
            print('[!] No faces detected.')
            return

        frame_draw = img.copy()
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

        output_path = os.path.join(self.output_dir, 'detected_image.jpg')
        frame_draw.save(output_path)
        print(f'[✓] Saved output to {output_path}')

    def _process_video(self):
        print('[+] Processing Video')
        try:
            video = mmcv.VideoReader(self.path)
        except Exception as e:
            print(f'[!] Failed to load video: {e}')
            return

        frames_tracked = []
        for i, frame in enumerate(video):
            print(f'\r[→] Tracking frame {i + 1}/{len(video)}', end='')
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, _ = self.mtcnn.detect(pil_frame)

            frame_draw = pil_frame.copy()
            draw = ImageDraw.Draw(frame_draw)
            if boxes is not None:
                for box in boxes:
                    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
            resized_frame = frame_draw.resize((640, 360), Image.BILINEAR)
            frames_tracked.append(resized_frame)

        print('\n[✓] Finished face tracking. Writing video...')

        output_path = os.path.join(self.output_dir, 'detected_video.mp4')
        dim = frames_tracked[0].size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, 25.0, dim)

        for frame in frames_tracked:
            out_video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        out_video.release()
        print(f'[✓] Saved video to {output_path}')
