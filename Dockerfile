FROM python:3.12-slim

# libGL, libglib 등 OpenCV가 필요로 하는 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# requirements.txt 복사 후 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# Gunicorn 실행 (app.py 안에 Flask app 객체 이름이 'app' 이라고 가정)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]