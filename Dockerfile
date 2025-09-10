# 1. 베이스 이미지: Python 3.12 슬림 버전 사용
FROM python:3.12

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements.txt 복사 및 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 나머지 앱 소스 코드 복사
COPY . .

# 5. gunicorn 실행을 위한 포트 노출
EXPOSE 5000

# 6. 애플리케이션 실행
CMD ["gunicorn", "--workers", "1", "--threads", "4", "-b", "0.0.0.0:5000", "app:app"]