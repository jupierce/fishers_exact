FROM debian
USER 0
RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-pip
RUN pip install cython fast-fisher django django-tables2 google-cloud-bigquery 'sqlalchemy>=1.1.9,<2.0.0' sqlalchemy-bigquery pyarrow tqdm 'google-cloud-bigquery[bqstorage,pandas]' pandas
COPY readiness/ readiness/
WORKDIR readiness
ENTRYPOINT python3 manage.py runserver 0.0.0.0:8000