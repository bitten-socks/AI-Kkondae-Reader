# 베이스 이미지
FROM python:3.12-slim

# 필수 시스템 패키지 설치 (OpenCV 종속성 포함)
# 필수 시스템 패키지 설치 (OpenCV 종속성 포함)
# 필수 시스템 패키지 설치 (OpenCV 의존성 포함)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


# 작업 디렉토리 설정
WORKDIR /app

# requirements 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Node 관련 파일 복사 (이미지 로그에서 node 관련이 있었음)
COPY package*.json yarn.lock* pnpm-lock.yaml* ./

# 앱 소스 복사
COPY --chown=python:python ./ ./

# 권한 재설정 및 git 제거
RUN chown -R python:python /app && rm -rf .git*

# 포트 노출
EXPOSE 5000

# gunicorn 실행
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]