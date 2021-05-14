docker build -t ml4qs:base .
docker run --rm -it -v "%cd%"/Python3Code:/root ml4qs:base /bin/bash