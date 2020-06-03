import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw


test_image_path = 'data/multi_face1.jpg'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
img = Image.open(test_image_path)

boxes, _ = mtcnn.detect(img)
frame_draw = img.copy()
draw = ImageDraw.Draw(frame_draw)
for box in boxes:
    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
frame_draw.show()
print(boxes)

class FaceDetector:
    def __init__(self):
        super().__init__()
    def process_video():
        print("I Process Videos bois")
    def process_frame():
        print("I Process frame bois")