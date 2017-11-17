
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
