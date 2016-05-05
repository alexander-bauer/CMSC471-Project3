.PHONY: all clean

all:

clean:
	rm -rf results/

env: requirements.txt
	python2 -m virtualenv env
	env/bin/pip install -r requirements.txt
	ln -sf /usr/lib/python2.7/site-packages/cv.py env/lib/python2.7/site-packages
	ln -sf /usr/lib/python2.7/site-packages/cv2.so env/lib/python2.7/site-packages

model.pkl:
	./proj3.py --train Data/training

%.pdf: %.tex
	pdflatex --halt-on-error $^
