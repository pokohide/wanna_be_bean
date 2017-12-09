import cv2
import dlib
import numpy

import glob
import os

PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)

class NoFaces(Exception):
    pass

class Face:
  def __init__(self, image, rect):
    self.image = image
    self.landmarks = numpy.matrix(
      [[p.x, p.y] for p in PREDICTOR(image, rect).parts()]
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

  COLOR_CORRECT_BLUR_FRAC = 0.7

  def __init__(self, before_after = True):
    self.detector = dlib.get_frontal_face_detector()
    self._load_beans()
    self.before_after = before_after

  def load_faces_from_image(self, image_path):
    """
      画像パスから画像オブジェクトとその画像から抽出した特徴点を読み込む。
      ※ 画像内に顔が1つないし複数検出された場合も、返すので正確には「特徴点配列」の配列を返す
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (image.shape[1] * self.SCALE_FACTOR,
                               image.shape[0] * self.SCALE_FACTOR))

    rects = self.detector(image, 1)

    if len(rects) == 0:
      raise NoFaces
    else:
      print("Number of faces detected: {}".format(len(rects)))

    faces = [Face(image, rect) for rect in rects]
    return image, faces

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
    original, faces = self.load_faces_from_image(image_path)

    # base_imageに合成していく
    base_image = original.copy()

    for face in faces:
      bean = self._get_bean_similar_to(face)
      bean_mask = self.get_face_mask(bean)

      M = self.transformation_from_points(
        face.landmarks[self.ALIGN_POINTS],
        bean.landmarks[self.ALIGN_POINTS]
      )

      warped_bean_mask = self.warp_image(bean_mask, M, base_image.shape)
      combined_mask = numpy.max(
        [self.get_face_mask(face), warped_bean_mask], axis = 0
      )

      warped_image = self.warp_image(bean.image, M, base_image.shape)
      warped_corrected_image = self.correct_colors(base_image, warped_image, face.landmarks)
      base_image = base_image * (1.0 - combined_mask) + warped_corrected_image * combined_mask

    path, ext = os.path.splitext( os.path.basename(image_path) )
    cv2.imwrite('outputs/output_' + path + ext, base_image)

    if self.before_after is True:
      before_after = numpy.concatenate((original, base_image), axis = 1)
      cv2.imwrite('before_after/' + path + ext, before_after)

  def _draw_convex_hull(self, image, points, color):
    "指定したイメージの領域を塗りつぶす"

    points = cv2.convexHull(points)
    cv2.fillConvexPoly(image, points, color = color)

  def _load_beans(self):
    "Mr. ビーンの画像をロードして、顔(特徴点など)を検出しておく"

    self.beans = []
    for image_path in glob.glob(os.path.join('beans', '*.jpg')):
      image, bean_face = self.load_faces_from_image(image_path)
      self.beans.append(bean_face[0])
    print('Mr. Beanをロードしました.')

  def _get_bean_similar_to(self, face):
    "特徴点の差分距離が小さいMr.ビーンを返す"

    get_distances = numpy.vectorize(lambda bean: numpy.linalg.norm(face.landmarks - bean.landmarks))

    distances = get_distances(self.beans)
    return self.beans[distances.argmin()]

if __name__ == '__main__':
  be_bean = BeBean()
  be_bean.to_bean('./images/kanna.jpg')
  # be_bean.to_bean('./images/kikuti.jpg')
  # be_bean.to_bean('./images/avengers.jpg')
  # be_bean.to_bean('./images/thor.jpg')
  # be_bean.to_bean('./images/x-men.jpg')
  # be_bean.to_bean('./images/iron-man.jpg')
  # be_bean.to_bean('./images/captain.jpg')

