FROM python:3.12-slim

WORKDIR /app

COPY src/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/train_model.py .

COPY src/main.py .

RUN python train_model.py

EXPOSE 8000

CMD ["fastapi", "run", "main.py"]
