import cv2
import dlib
import numpy

import glob
import os

PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)

class Face:
  def __init__(self, image, rect):
    self.image = image
    self.rect = rect
    self.landmarks = numpy.matrix(
      [[p.x, p.y] for p in PREDICTOR(image, self.rect).parts()]
    )

class BeBean:
  SCALE_FACTOR = 1
  FEATHER_AMOUNT = 11

  # 特徴点のうちそれぞれの部位を表している配列のインデックス
  FACE_POINTS = list(range(17, 68))
  MOUTH_POINTS = list(range(48, 61))
  RIGHT_BROW_POINTS = list(range(17, 22))
  LEFT_BROW_POINTS = list(range(22, 27))
  RIGHT_EYE_POINTS = list(range(36, 42))
  LEFT_EYE_POINTS = list(range(42, 48))
  NOSE_POINTS = list(range(27, 35))
  JAW_POINTS = list(range(0, 17))

  ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS +
    NOSE_POINTS + MOUTH_POINTS)

  # オーバーレイする特徴点
  OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS]

  COLOR_CORRECT_BLUR_FRAC = 0.6

  def __init__(self):
    self.detector = dlib.get_frontal_face_detector()
    self._load_beans()

  def load_faces_from_image(self, image_path):
    """
      画像パスから画像オブジェクトとその画像から抽出した特徴点を読み込む。
      ※ 画像内に顔が1つないし複数検出された場合も、返すので正確には「特徴点配列」の配列を返す
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (image.shape[1] * self.SCALE_FACTOR,
                               image.shape[0] * self.SCALE_FACTOR))

    rects = self.detector(image, 1)
    if len(rects) == 0: raise NoFaces
    return [Face(image, rect) for rect in rects]

  def transformation_from_points(self, t_points, o_points):
    """
      特徴点から回転やスケールを調整する。
      t_points: (target points) 対象の特徴点(入力画像)
      o_points: (origin points) 合成元の特徴点(つまりビーン)
    """

    t_points = t_points.astype(numpy.float64)
    o_points = o_points.astype(numpy.float64)

    t_mean = numpy.mean(t_points, axis = 0)
    o_mean = numpy.mean(o_points, axis = 0)

    t_points -= t_mean
    o_points -= o_mean

    t_std = numpy.std(t_points)
    o_std = numpy.std(o_points)

    t_points -= t_std
    o_points -= o_std

    # 行列を特異分解しているらしい
    # https://qiita.com/kyoro1/items/4df11e933e737703d549
    U, S, Vt = numpy.linalg.svd(t_points.T * o_points)
    R = (U * Vt).T

    return numpy.vstack(
      [numpy.hstack((( o_std / t_std ) * R, o_mean.T - ( o_std / t_std ) * R * t_mean.T )),
      numpy.matrix([ 0., 0., 1. ])]
    )


  def get_face_mask(self, face):
    image = numpy.zeros(face.image.shape[:2], dtype = numpy.float64)
    for group in self.OVERLAY_POINTS:
      self._draw_convex_hull(image, face.landmarks[group], color = 1)

    image = numpy.array([ image, image, image ]).transpose((1, 2, 0))
    image = (cv2.GaussianBlur(image, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
    image = cv2.GaussianBlur(image, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)

    return image

  def warp_image(self, image, M, dshape):
    output_image = numpy.zeros(dshape, dtype = image.dtype)
    cv2.warpAffine(
      image,
      M[:2],
      (dshape[1], dshape[0]),
      dst = output_image, borderMode = cv2.BORDER_TRANSPARENT, flags = cv2.WARP_INVERSE_MAP
    )
    return output_image

  def correct_colors(self, t_image, o_image, t_landmarks):
    """
      対象の画像に合わせて、色を補正する
    """
    blur_amount = self.COLOR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
      numpy.mean(t_landmarks[self.LEFT_EYE_POINTS], axis = 0) -
      numpy.mean(t_landmarks[self.RIGHT_EYE_POINTS], axis = 0)
    )
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0: blur_amount += 1

    t_blur = cv2.GaussianBlur(t_image, (blur_amount, blur_amount), 0)
    o_blur = cv2.GaussianBlur(o_image, (blur_amount, blur_amount), 0)

    # ゼロ除算を避ける　
    o_blur += (128 * (o_blur <= 1.0)).astype(o_blur.dtype)

    return (o_image.astype(numpy.float64) * t_blur.astype(numpy.float64) / o_blur.astype(numpy.float64))

  def to_bean(self, image_path):
    faces = self.load_faces_from_image(image_path)
    for face in faces:
      # print(face.landmarks)
      # print(len(face.landmarks))
      # print('---')
      # print((numpy.linalg.norm(self.beans[0].landmarks - face.landmarks)))
      bean = self.beans[0]

      M = self.transformation_from_points(
        face.landmarks[self.ALIGN_POINTS],
        bean.landmarks[self.ALIGN_POINTS]
      )

      bean_mask = self.get_face_mask(bean)
      warped_bean_mask = self.warp_image(bean_mask, M, face.image.shape)
      combined_mask = numpy.max(
        [self.get_face_mask(face), warped_bean_mask], axis = 0
      )

      warped_image = self.warp_image(bean.image, M, face.image.shape)
      warped_corrected_image = self.correct_colors(face.image, warped_image, face.landmarks)
      output_image = face.image * (1.0 - combined_mask) + warped_corrected_image * combined_mask
    cv2.imwrite('output.jpg', output_image)

  def _draw_convex_hull(self, image, points, color):
    "指定したイメージの領域を塗りつぶす"

    points = cv2.convexHull(points)
    cv2.fillConvexPoly(image, points, color = color)

  def _load_beans(self):
    "Mr. ビーンの画像をロードして、顔(特徴点など)を検出しておく"

    self.beans = []
    for image_path in glob.glob(os.path.join('beans', '*.png')):
      bean_face = self.load_faces_from_image(image_path)[0]
      self.beans.append(bean_face)
    print('Mr. Beanをロードしました.')

if __name__ == '__main__':
  be_bean = BeBean()
  # be_bean.to_bean('./images/thor.jpg')
  # be_bean.to_bean('./images/x-men.jpg')
  be_bean.to_bean('./images/iron-man.jpg')
  # be_bean.to_bean('./images/captain.jpg')

