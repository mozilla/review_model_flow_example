# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

FROM --platform=linux/amd64 python:3.10.11-bullseye

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

COPY requirements.outerbounds.txt requirements.outerbounds.txt

RUN python -m pip install -r requirements.txt 
