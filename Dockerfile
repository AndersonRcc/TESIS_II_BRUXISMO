FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/site/wwwroot

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "app:app", "--bind=0.0.0.0:8000", "--timeout", "300"]