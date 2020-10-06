FROM python:3.6-slim

COPY ./webservice /webservice

COPY docker-requirements.txt /webservice

# updating the machine
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install build-essential -y
RUN apt-get install python3-pip -y

# Creating python environment for the app
RUN pip3 install -r /webservice/docker-requirements.txt
