# Use Python 3.10 and Node 20 as base
FROM nikolaik/python-nodejs:python3.10-nodejs20

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/package*.json ./backend/
RUN cd backend && npm install

# Install Python requirements
RUN pip install --no-cache-dir ultralytics opencv-python-headless pymupdf numpy

# Copy application code
COPY ai/ ./ai/
COPY backend/ ./backend/

# Expose backend API port
EXPOSE 5000

# Set environment to production
ENV NODE_ENV=production

# Start Node.js server
CMD ["node", "backend/index.js"]
