FROM python:3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    build-essential \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libpostproc-dev \ 
    gcc \
    make \
    portaudio19-dev \
    python3-pyaudio \
    python3-all-dev \
    git

WORKDIR /app

# Copy git configuration files first
COPY .gitmodules ./
COPY .git ./.git

# Initialize and update submodules
RUN git submodule init && git submodule update --recursive

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY ./dimos ./dimos
COPY ./tests ./tests
COPY ./dimos/__init__.py ./

CMD [ "python", "-m", "dimos.robot.unitree.unitree_go2" ]
