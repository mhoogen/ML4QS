FROM ubuntu:20.04
FROM python:3.8.8

RUN apt-get update
RUN apt-get install sudo
RUN apt-get install git -y


COPY requirements.txt /src/requirements.txt

RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip
RUN pip3 install Cython
RUN pip3 install python-dev-tools

RUN xargs -L 1 pip3 install < /src/requirements.txt

WORKDIR /root
RUN python3 --version

