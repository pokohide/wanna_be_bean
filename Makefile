init:
	wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
	bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

run:
	python3 be_bean.py
