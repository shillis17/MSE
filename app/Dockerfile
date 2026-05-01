FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl unzip wget && rm -rf /var/lib/apt/lists/*

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN mkdir -p /app/data

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]