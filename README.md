# EE 782: AI Guard Agent

## Introduction
The aim of this assignment is to create a room guarding agent which utilizes pre-trained models to identify the intruder and prevent further breach of privacy based on particular escalation scenarios

## How to use
1. Install python 3.11 because this is a requirement for using face recognition module. The link to download the same can be found [here](https://www.python.org/downloads/release/python-3118).
2. Download FFmpeg. The steps to do the same can be found in this [video](https://youtu.be/K7znsMo_48I?si=-HXmmoZVc5bOmCEC).
3. Create a new viertual environment using the command: python -m venv ai_guard_env ( Can also be done using Conda)
4. Activate the environment
5. Install dependencies using the following steps:
    i)   pip install --upgrade pip
    ii)  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    iii) pip install -r requirements.txt
   Note: If you are unable to install the dependency dlib, download the buildwheel file from the repo and directly install it using : pip install C:\filepath\dlib-19.24.1-cp311-cp311-win_amd64.whl
6. If you want to register your face then you will have to run the Face_recognition.py file
7. If you wish to use the Guard agent then you will have to run the AI_guard_integrated.py

In case you have any doubts on the parts I have done just message me directly and dont waste your time trying to figure it out yourself. 
The next step would be to integrate the systems and also work on the escalation logic for the LLM
