# Use the official PyTorch image compatible with AMD64
FROM --platform=linux/amd64 pytorch/pytorch:2.1.0-cuda11.7-cudnn8-runtime

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

# Create the program directory
RUN mkdir -p /opt/program

COPY constraints.txt .

# Install FastAPI, Uvicorn, librosa, torch, and transformers
RUN pip3 install fastapi==0.112.0 
RUN pip3 install uvicorn==0.30.5 
RUN pip3 install librosa==0.10.2 
RUN pip3 install transformers==4.44.0
RUN pip3 install pydub==0.25.1
RUN pip3 install pyannote.audio==3.3.0 -c constraints.txt

# Use a build argument to force cache invalidation for the FastAPI app copy
ARG CACHEBUST=1

# Copy your FastAPI app with cache busting
COPY --chown=root:root fastapi_app /opt/program/fastapi_app

# Copy the serve script
COPY serve /usr/local/bin/serve

# Make the serve script executable
RUN chmod +x /usr/local/bin/serve

# Set the working directory
WORKDIR /opt/program

EXPOSE 8080

# Define the entry point
ENTRYPOINT ["serve"]