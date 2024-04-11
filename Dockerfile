FROM continuumio/miniconda3 as parastell-deps

ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update -y
RUN apt-get upgrade -y

# install dependencies
RUN apt-get install -y libgl1-mesa-glx \
                        libgl1-mesa-dev \
                        libglu1-mesa-dev \
                        freeglut3-dev \
                        libosmesa6 \
                        libosmesa6-dev \
                        libgles2-mesa-dev \
                        curl \
                        wget \
                        libx11-6 \
                        libxt6 \
                        libgl1 \
                        libxcb-icccm4 \
                        libxcb-image0 \
                        libxcb-keysyms1 \
                        libxcb-render-util0 \
                        libxkbcommon-x11-0 \
                        libxcb-randr0 \
                        libxcb-xinerama0 \
                        libxm4 \
                        libtiff5 \
                        libxcursor1 \
                        libxinerama1

# download cubit
RUN wget -O /cubit.deb https://f002.backblazeb2.com/file/cubit-downloads/Coreform-Cubit/Releases/Linux/Coreform-Cubit-2023.11%2B43088-Lin64.deb

# install cubit
RUN dpkg -i cubit.deb
ENV PYTHONPATH=/opt/Coreform-Cubit-2023.11/bin/
COPY ./rlmcloud.in /opt/Coreform-Cubit-2023.11/bin/licenses/rlmcloud.in

RUN mkdir -p /opt/etc
RUN cp /root/.bashrc /opt/etc/bashrc

# parastell env
COPY ./environment.yml /environment.yml
RUN conda env create -f environment.yml
RUN echo "conda activate parastell_env" >> /opt/etc/bashrc

WORKDIR /opt

# install pystell_uw
RUN git clone https://github.com/aaroncbader/pystell_uw.git
ENV PYTHONPATH=$PYTHONPATH:/opt/pystell_uw

WORKDIR /

from parastell-deps as parastell

# install parastell
RUN git clone https://github.com/svalinn/parastell.git && \
    cd parastell && \
    git checkout gh_action
ENV PYTHONPATH=$PYTHONPATH:/opt/parastell


