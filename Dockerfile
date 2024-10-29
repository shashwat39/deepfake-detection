# Use a base image with Python support
FROM python:3.9-slim

# Create and set the working directory
WORKDIR /app

# Copy the contents of the local directory to the container’s working directory
COPY . /app

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements_inference.txt

# Expose the port where the FastAPI app will run
EXPOSE 8000

# Define the command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
