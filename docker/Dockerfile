FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common wget git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update

RUN apt-get install -y --no-install-recommends \
        python3.7 \
        python3.7-dev \
        python3-pip

RUN ln -s /usr/bin/python3.7 /usr/local/bin/python3

RUN apt-get install -y zsh && \
    chsh -s /bin/zsh && \
    zsh -c "$(wget https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"

ADD . /app
WORKDIR /app

RUN apt-get install -y $(awk '{print $1}' docker/apt.packages)

RUN python3 -m pip install -U pip setuptools wheel
RUN python3 -m pip install -r ./requirements.txt

CMD exec /bin/zsh -c "trap : TERM INT; sleep infinity & wait"
