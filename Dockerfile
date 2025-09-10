# 1. 베이스 이미지: 모든 라이브러리가 포함된 표준 Python 이미지 사용
FROM python:3.12

# 2. 시스템 라이브러리 설치: 바뀐 이름으로 수정!
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    cmake \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. requirements.txt 복사 및 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 나머지 소스 코드 전체 복사
COPY . .

# 6. 애플리케이션 실행 (gunicorn)
CMD ["gunicorn", "--workers", "1", "--threads", "4", "-b", "0.0.0.0:5000", "app:app"]