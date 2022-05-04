# prod dockerfile
# 1. Base image
FROM python:3.8-slim

WORKDIR /app

# here we create a new user
# note how the commands are using &&
# this helps with caching
RUN useradd -m -r user && \
    chown user /app

# 2. Copy requirements.txt
COPY requirements.txt /app

# 3. Install dependencies
RUN pip install -r requirements.txt
USER user

# copy source files
COPY src /app

# lazy network access
# EXPOSE 5005

CMD ["python","main.py"]
