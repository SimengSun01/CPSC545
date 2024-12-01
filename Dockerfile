# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set non-interactive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies including Python3 and pip
RUN apt-get update && apt-get install -y \
    curl \
    bzip2 \
    openjdk-11-jre-headless \
    git \
    python3 \
    python3-pip \
    && apt-get clean

# Install Python libraries
RUN pip3 install pandas numpy scikit-learn matplotlib seaborn scanpy

# Create a symlink so both python and python3 commands are available
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Nextflow
RUN curl -fsSL https://get.nextflow.io | bash && \
    mv nextflow /usr/local/bin/ && \
    chmod +x /usr/local/bin/nextflow

# Add /usr/local/bin to PATH explicitly (if needed)
ENV PATH="/usr/local/bin:$PATH"

# Set working directory and copy files
WORKDIR /app
COPY . /app/

# Default command to run Nextflow
CMD ["nextflow", "run", "test.nf"]

