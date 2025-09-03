FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies in isolated environment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

EXPOSE 5001

CMD ["python", "app.py"]