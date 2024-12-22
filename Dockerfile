FROM python:3.10-slim

WORKDIR /app

COPY api /app/
COPY artifacts/objects.pkl /app/
COPY requirements.txt /app/
COPY src/data_transformation.py /app/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt_tab punkt stopwords

EXPOSE 80

CMD ["fastapi", "run", "app.py", "--port", "80"]