FROM ubuntu:18.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
 && rm -rf /var/lib/apt/lists/*

RUN apt -y update && apt -y install wget curl vim libgcrypt20 coreutils libgl1-mesa-glx
RUN wget https://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ./miniconda.sh
RUN chmod ouga+xw ./miniconda.sh
RUN bash ./miniconda.sh -b -p ./miniconda

 # Create a working directory
RUN conda install pytorch-cpu torchvision-cpu -c pytorch
RUN conda install -y --file requirements.txt

EXPOSE 8080

CMD ["gunicorn", , "app:app"]