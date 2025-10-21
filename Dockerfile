# Use an official Python runtime as a parent image
FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure logs show in real time
ENV PYTHONUNBUFFERED=1

# Prepare the data directory
CMD ["python", "data_processing.py"]

# Run experiments
# CMD ["python", "run_base_prompt.py"]
# CMD ["python", "run_improved_prompt.py"]
# CMD ["python", "run_few_shot_prompt.py"]
# CMD ["python", "run_opro_prompt.py"]