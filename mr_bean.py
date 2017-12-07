# -*- coding:utf-8 -*-

import numpy
import cv2
from PIL import Image

class Beany:

  # CASCADE_PATH = './haarcascades/haarcascade_frontalface_alt.xml'
  CASCADE_PATH = './haarcascades/haarcascade_frontalface_alt2.xml'
  # CASCADE_PATH = './haarcascades/haarcascade_frontalface_alt_tree.xml'
  # CASCADE_PATH = './haarcascades/haarcascade_frontalface_default.xml'
  EYE_CASCADE_PATH = './haarcascades/haarcascade_eye.xml'
  SMILE_CASCADE_PATH = './haarcascades/haarcascade_smile.xml'
  RED = (0, 0, 200)
  BLUE = (255, 0, 0)
  WHITE = (0, 0, 0)
  BLACK = (255, 255, 255, 0)

  def __init__(self):
    self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)
    self.eye_cascade = cv2.CascadeClassifier(self.EYE_CASCADE_PATH)

  def beanalize(self, image_path):
    "画像をビーン化させる"

    image = cv2.imread(image_path)
    self._check(image)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 顔領域の探索
    face_rect = self.face_cascade.detectMultiScale(
      image_gray, scaleFactor = 1.1, minNeighbors = 1, minSize = (10, 10))

    print(len(face_rect))

    # 顔領域を赤色の短形で囲む
    if len(face_rect) <= 0: return

    self.compound(image_path, face_rect)
    return

    for (x, y, w, h) in face_rect:
      cv2.rectangle(image, (x, y), (x + w, y + h), self.RED, thickness = 3)
      face_image_gray = image_gray[y: y + h, x: x + w]
      face_image_color = image[y: y + h, x: x + w]
      eyes = self.eye_cascade.detectMultiScale(
        face_image_gray, scaleFactor = 1.1, minNeighbors = 2, minSize = (1, 1))
      for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face_image_color, (ex, ey), (ex + ew, ey + eh), self.BLUE, thickness = 3)

    cv2.imwrite('result.jpg', image)

  def compound(self, image_path, face_rect):
    source = Image.open(image_path).convert('RGBA')
    layer = Image.open('./beans/4.png')

    canvas = Image.new('RGBA', source.size, self.BLACK)

    for (x, y, w, h) in face_rect:
      resized_image = layer.resize((w, h), Image.ANTIALIAS)
      canvas.paste(resized_image, (x, y), resized_image)

    result = Image.alpha_composite(source, canvas)
    result.save('result.png', format = 'png')

  def _get_similar_bean(self, eyes):
    "目の傾きから似ているビーンを取得する"

  def _check(self, image):
    "画像が存在するかチェック。ない場合は終了させる。"
    if image is None:
      print('This image colud not be opened.')
      quit()

if __name__ == '__main__':
  beany = Beany()
  # beany.beanalize('./images/x-men.jpg')
  # beany.beanalize('./images/iron-man.jpg')
  beany.beanalize('./images/captain.jpg')

