FROM ubuntu:latest

RUN apt-get update
RUN apt-get install sudo
RUN apt-get install git -y

ADD Python3_requirements.txt /src/requirements.txt

RUN apt-get install python3-pip -y
RUN pip3 install Cython

RUN xargs -L 1 pip3 install < /src/requirements.txt

WORKDIR /root

RUN python3 --version
