docker build -t ml4qs:base .
docker run --rm -it -v "%cd%":/root ml4qs:base /bin/bash
