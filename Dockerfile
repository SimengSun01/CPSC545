#base image
FROM ubuntu:20.04
#avoid prompts (for now)
ENV DEBIAN_FRONTEND=noninteractive

# python3 and pip
RUN apt-get update && apt-get install -y \
    curl \
    bzip2 \
    openjdk-11-jre-headless \
    git \
    python3 \
    python3-pip \
    && apt-get clean

########
RUN pip3 install pandas numpy scikit-learn matplotlib seaborn scanpy
#symlink 
RUN ln -s /usr/bin/python3 /usr/bin/python

#install Nextflow
RUN curl -fsSL https://get.nextflow.io | bash && \
    mv nextflow /usr/local/bin/ && \
    chmod +x /usr/local/bin/nextflow


ENV PATH="/usr/local/bin:$PATH"
WORKDIR /app
COPY . /app/
CMD ["nextflow", "run", "test.nf"]

