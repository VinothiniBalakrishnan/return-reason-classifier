FROM python:3.10-slim
WORKDIR /app
RUN useradd -m -u 1000 appuser
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=appuser:appuser app.py classifier.py processing.py ./
COPY --chown=appuser:appuser data ./data
USER appuser
ENV HOME=/home/appuser PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--browser.gatherUsageStats=false", "--server.maxUploadSize=10"]
