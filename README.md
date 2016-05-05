# Use and Care

Use of this project requires a pre-installed copy of OpenCV 2 or 3 with Python 2 bindings.
Other requirements are listed in `requirements.txt`, and may be installed in a virtual environment
using `make env`. If using a Python installed outside of a virtual environment, instead of invoking
`proj3.py` as an executable, invoke your preferred Python as `python proj3.py [args]`.

If a pre-trained model is present, as in this repository, the program `proj3.py` will function and
classify any images it is passed on the command line. If the model is not present, it can be invoked
with `proj3.py --train Data/training` to train and test a new one.

To classify a single image, use `proj3.py path/to/image`. Multiple images or directories may be
supplied. The resulting class will be printed.
