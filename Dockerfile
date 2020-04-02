FROM ubuntu:latest

RUN apt-get update
RUN apt-get install sudo
RUN apt-get install python3-pip -y

# WORKDIR /root/PythonCode

RUN python3 --version

