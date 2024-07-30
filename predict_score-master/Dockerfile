FROM python:3.9

workdir /prediction_model

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD ['streamlilt run prediction_model.py',]