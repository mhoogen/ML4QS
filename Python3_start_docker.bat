docker build -t ml4qs:base .
docker run --rm -it -v "%cd%":/root/Python3Code ml4qs:base /bin/bash