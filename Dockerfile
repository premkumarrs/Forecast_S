FROM python:3.11-slim-bookworm

RUN mkdir -p app
COPY . /app/.
WORKDIR /app

RUN pip3 install cython
RUN pip3 install -r requirements.txt --no-cache-dir

# REENABLE WHEN TESTS ARE DONE
# RUN useradd -ms /bin/bash appuser
# USER appuser
