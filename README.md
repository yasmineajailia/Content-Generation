#HOW TO RUN

#create venv

python -m venv .venv

#activate venv

.venv\Scripts\activate

#install requirements

pip install -r requirements.txt

#Set up environment variables for model storage

set HF_HOME=D:\HuggingFace

set TRANSFORMERS_CACHE=D:\HuggingFace\transformers

set HF_DATASETS_CACHE=D:\HuggingFace\datasets

#Create necessary directories:

mkdir D:\HuggingFace

mkdir D:\HuggingFace\transformers

mkdir D:\HuggingFace\datasets

mkdir data

mkdir models

mkdir generated_gifs

#run application

python app.py


