FROM continuumio/miniconda3

ENV TZ=America/Chicago
ENV PYTHONPATH=/opt/Coreform-Cubit-2024.3/bin/
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt-get install -y libgl1-mesa-glx \
                       libgl1-mesa-dev \
                       libglu1-mesa-dev \
                       freeglut3-dev \
                       libosmesa6 \
                       libosmesa6-dev \
                       libgles2-mesa-dev \
                       curl \
                       wget \
                       libxm4 
RUN apt-get clean

# download cubit
RUN wget -O /cubit.deb https://f002.backblazeb2.com/file/cubit-downloads/Coreform-Cubit/Releases/Linux/Coreform-Cubit-2024.3%2B46968-Lin64.deb


# install dependencies
RUN apt-get install -y libx11-6 
RUN apt-get install -y libxt6 
RUN apt-get install -y libgl1
RUN apt-get install -y libglu1-mesa
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libxcb-icccm4 
RUN apt-get install -y libxcb-image0 
RUN apt-get install -y libxcb-keysyms1 
RUN apt-get install -y libxcb-render-util0 
RUN apt-get install -y libxkbcommon-x11-0 
RUN apt-get install -y libxcb-randr0 
RUN apt-get install -y libxcb-xinerama0

# install cubit
RUN dpkg -i cubit.deb

