import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2, mmcv
from PIL import Image, ImageDraw
import numpy as np


class FaceDetector:
    def __init__(self, input_path, input_type='image'):
        self.path = input_path
        self.type = input_type
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def process(self):
        if self.type == 'image':
            print('[+] Processing Image')
            img = Image.open(self.path)
            boxes, _ = self.mtcnn.detect(img)
            frame_draw = img.copy()
            draw = ImageDraw.Draw(frame_draw)
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
            frame_draw.save('output.jpg')
        elif self.type == 'video':
            print('[+] Processing Video')
            frames_tracked = []
            video = mmcv.VideoReader(self.path)
            frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

            for i, frame in enumerate(frames):
                print('\rTracking frame: {}'.format(i + 1), end='')
                boxes, _ = self.mtcnn.detect(frame)

                frame_draw = frame.copy()
                draw = ImageDraw.Draw(frame_draw)
                if type(boxes) != type(None):
                    for box in boxes:
                        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
                frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
            dim = frames_tracked[0].size
            fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim)
            for frame in frames_tracked:
                video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
            video_tracked.release()
        else:
            print('Error: Invalid Input Type')


if __name__ == '__main__':
    # facedetect = FaceDetector('data/single_face1.jpg')
    # facedetect.process()
    facedetect = FaceDetector('data/video_test.mp4', 'video')
    facedetect.process()
