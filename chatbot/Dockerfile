FROM python:3.9.6-slim

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

EXPOSE 8080

RUN pip install transformers==4.2.2
RUN pip install --no-deps sentence-transformers==1.2.1
RUN pip install gunicorn
RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 lineBot:app
# CMD python manage.py runserver 0.0.0.0:8080