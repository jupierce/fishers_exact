FROM debian
USER 0
RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-pip
RUN pip install cython fast-fisher django django-tables2 google-cloud-bigquery
COPY readiness/ readiness/
WORKDIR readiness
ENTRYPOINT python3 manage.py runserver 0.0.0.0:8000