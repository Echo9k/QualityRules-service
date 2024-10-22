# Base image with Python
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the Ray Serve port (e.g., 8000)
EXPOSE 8000

# Run the server
CMD ["python", "serve.py"]
