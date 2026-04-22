# Use a specific, stable version of Python
FROM python:3.11-slim-bookworm

# 1. Install system dependencies and Rust toolchain
# tiktoken requires Rust to compile from source
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && apt-get clean

# 2. Add Rust/Cargo to the system PATH
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# 3. Copy only requirements first to leverage Docker layer caching
# This prevents re-installing all packages every time you change a line of code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
COPY . .

# 5. Streamlit defaults to 8501; make sure this matches your CMD
EXPOSE 8501

# 6. Run Streamlit and bind it to 0.0.0.0 so it's accessible outside the container
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]