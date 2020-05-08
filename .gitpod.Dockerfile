 
FROM gitpod/workspace-full
                    
USER gitpod

RUN pip3 install torch torchvision
RUN pip3 install tqdm
RUN pip install torchsummary
RUN pip install torchviz


# Install custom tools, runtime, etc. using apt-get
# For example, the command below would install "bastet" - a command line tetris clone:
#
# RUN sudo apt-get -q update && #     sudo apt-get install -yq bastet && #     sudo rm -rf /var/lib/apt/lists/*
#
# More information: https://www.gitpod.io/docs/config-docker/
