
https://twitter.com/Wordsworth923/status/912828390662561792

このツイートが楽しかったから、これを地味に実現するものを作りたい。

Pythonで色々な画像から顔認識をして、その顔の部分にいい感じのMr.ビーンの顔を合成する。

合成元画像は「アメコミ　映画」とGoogle画像検索した結果の画像から顔を認識して、いい感じのミスタービーンの顔を合成する。

- ミスタービーンの顔だけを抽出する(手作業 png形式)
- 顔の検出
- 顔の向き?目や鼻の位置? などの情報を元に近いミスタービーンの画像を探す。
- ミスタービーンのpng画像を透過合成

https://qiita.com/k_sui_14/items/bb9dc8395da85400e518
https://blanktar.jp/blog/2015/02/python-opencv-realtime-lauhgingman.html
http://symfoware.blog68.fc2.com/blog-entry-1557.html


## 手順

1. OpenCVをインストール(Mac)
https://blog.ymyzk.com/2015/07/os-x-opencv-3-python-2-3/

2. Macに`dlib`をインストール
https://qiita.com/matsu_mh/items/7955e9b1f14dc92a38fe

3. `shape_predictor_68_face_landmarks.dat`を[ダウンロード](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

## 課題
- 目の位置をとってその角度に近いビーンを選択
- ビーンが圧縮されないように比率を保ったままリサイズ
- 合成元の色にビーンの色を近づけてから合成

特徴点から似ているMr. ビーンを検出するのに、OpenCVの総当りマッチングアルゴリズムを採用する?
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_feature2d/py_matcher/py_matcher.html

dlibを使って顔認識とかをいい感じにやってくれる`face_recognition`というライブラリでの、特徴点の距離は普通に特徴点の位置の差分のノルムで計算していたので、
今回も特に頑張らず、単純に距離を測定して小さなものを選択しました。

## 参考記事

- [Switching Eds: Face swapping with Python, dlib, and OpenCV](https://matthewearl.github.io/2015/07/28/switching-eds-with-python/)
- [Face Swap on GitHub](https://github.com/hrastnik/FaceSwap)
- [Face morpher on GitHub](https://github.com/alyssaq/face_morpher)
- [OpenCVで平均顔を作るチュートリアル](https://medium.com/@NegativeMind/opencv%E3%81%A7%E5%B9%B3%E5%9D%87%E9%A1%94%E3%82%92%E4%BD%9C%E3%82%8B%E3%83%81%E3%83%A5%E3%83%BC%E3%83%88%E3%83%AA%E3%82%A2%E3%83%AB-94c48a5cd1f8)

https://github.com/mrgloom/Face-Swap
https://github.com/YuvalNirkin/face_swap
