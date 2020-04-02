docker build -t ml4qs:base .
docker run --rm -it -v %cd%:/root/PythonCode ml4qs:base /bin/bash