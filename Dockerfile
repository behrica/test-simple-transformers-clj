FROM clojure:openjdk-14-lein-2.9.1
RUN apt-get update && apt-get install -y wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN ./Miniconda3-latest-Linux-x86_64.sh -b
RUN /root/miniconda3/condabin/conda create -y -n transformers python==3.8.2 pandas tqdm
RUN /root/miniconda3/condabin/conda install -y -n transformers pytorch cpuonly -c pytorch
RUN /root/miniconda3/envs/transformers/bin/pip install simpletransformers torchvision
RUN mkdir /app
WORKDIR /app
COPY test.py /app
RUN /root/miniconda3/envs/transformers/bin/python test.py ## testing that simpletransformers is installed and works
COPY project.clj /app
RUN ["lein", "deps"]
COPY . /app
CMD ["lein","exec","-ep","(use 'test-simple-transformers-clj.core)"]
