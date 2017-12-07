# -*- coding:utf-8 -*-

import numpy
import os
import glob
import cv2
import dlib
import math
import face_recognition
from PIL import Image

class Beany:
  RED = (0, 0, 200)
  BLUE = (255, 0, 0)
  WHITE = (0, 0, 0)
  BLACK = (255, 255, 255, 0)

  def __init__(self):
    # self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    self._load_beans()

  def beanalize(self, image_path):
    "画像をビーン化させる"
    image = face_recognition.load_image_file(image_path)
    image_encodings = face_recognition.face_encodings(image)[0]

    self._get_similar_bean(image_encodings)
    return

    face_locations = face_recognition.face_locations(image)
    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    self._get_face_angle(image)
    return

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()
    return

    image = cv2.imread(image_path)
    self._check(image)
    image = self.fece_detector(image)
    cv2.imwrite('output_result.jpg', image)

  def _get_face_angle(self, image):
    face_landmarks = face_recognition.face_landmarks(image)
    for face_landmark in face_landmarks:
      left_xy = self._get_the_center(face_landmark['left_eye'])
      right_xy = self._get_the_center(face_landmark['right_eye'])

      # .mean(axis = (0, 1, 2))
      print(left_xy, right_xy)
      print(self._get_angle(left_xy, right_xy))
      print('---')

  def _get_angle(self, left_xy, right_xy):
    lx, ly = left_xy
    rx, ry = right_xy
    rad = math.atan2(ry - ly, rx - lx)
    deg = math.degrees(rad)
    return deg

  def _get_the_center(self, xys):
    total_x = 0
    total_y = 0
    for xy in xys:
      x, y = xy
      total_x += x
      total_y += y
    return (int(total_x / len(xys)), int(total_y / len(xys)))

  def _get_similar_bean(self, image_encodings):
    "特徴点からマッチングを行う"
    face_distances = face_recognition.face_distance(self.bean_encodins, image_encodings)
    similar = self.bean_encodins[0]
    for i, face_distance in enumerate(face_distances):
      # 似ているほど距離は短くなる
      print("The test image has a distance of {:.8} from known image #{}".format(face_distance, i))

  def _load_beans(self):
    "ビーンの画像をロードする"
    beans = []
    bean_encodins = []
    for file_path in glob.glob(os.path.join('beans', '*.png')):
      file = face_recognition.load_image_file(file_path)
      beans.append(file)
      face_encodings = face_recognition.face_encodings(file)[0]
      bean_encodins.append(face_encodings)
    print('Mr. Beanの特徴点をロードしました.')
    self.beans = beans
    self.bean_encodins = bean_encodins

  def _check(self, image):
    "画像が存在するかチェック。ない場合は終了させる。"
    if image is None:
      print('This image colud not be opened.')
      quit()

if __name__ == '__main__':
  beany = Beany()
  beany.beanalize('./images/thor.jpg')
  # beany.beanalize('./images/x-men.jpg')
  # beany.beanalize('./images/iron-man.jpg')
  # beany.beanalize('./images/captain.jpg')

