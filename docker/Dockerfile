FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY models/ models/

ENV PYTHONPATH=/app
ENV FLASK_ENV=production

EXPOSE 5001

CMD ["gunicorn", \
     "--bind", "0.0.0.0:5001", \
     "--workers", "2", \
     "--timeout", "60", \
     "src.api.app:app"]
