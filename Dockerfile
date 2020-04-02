FROM ubuntu:latest

RUN apt-get update
RUN apt-get install python3-pip -y

RUN pip3 install numpy pandas scipy matplotlib scikit-learn
RUN pip3 install gensim nltk unidecode statsmodels pykalman pyclust inspyred treelib

# risky installs
# RUN pip3 pybrain pyflux

# WORKDIR /root/PythonCode

RUN python3 --version

