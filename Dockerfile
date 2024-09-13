FROM python:3.10-slim
EXPOSE 8080

WORKDIR /app

COPY . ./



RUN pip install --no-cache-dir -r requirements.txt
RUN pip install tesseract
RUN pip install streamlit

ENTRYPOINT ["streamlit","run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]