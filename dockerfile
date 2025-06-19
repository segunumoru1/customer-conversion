# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Expose the Streamlit and FastAPI ports
EXPOSE 8501
EXPOSE 8000

# Run the Streamlit app when the container starts
CMD ["streamlit", "run", "src/app.py", "--server.port=8501"]

# For FastAPI, use:
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

