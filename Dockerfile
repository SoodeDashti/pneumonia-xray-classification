# Chest X-Ray Pneumonia Classifier - Gradio dashboard
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching - only reinstalls
# when requirements.txt actually changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and the trained model
COPY app.py .
COPY models/ ./models/

# Using 8860 instead of Gradio's default 7860 to avoid clashing with
# other Docker projects on this machine (port is also set in app.py)
EXPOSE 8860

CMD ["python", "app.py"]