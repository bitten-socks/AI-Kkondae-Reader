
from flask import Flask, request, jsonify
from flask_cors import CORS  # ⬅️ 1. flask-cors 불러오기
import cv2
import numpy as np
import base64
import io
from PIL import Image
import random
import mediapipe as mp # mediapipe import

app = Flask(__name__)
CORS(app)  # ⬅️ 2. CORS를 앱에 적용하여 모든 외부 요청을 허용

# dlib 얼굴 탐지기와 랜드마크 예측기 초기화
# Mediapipe 얼굴 인식 모델 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# 콘텐츠 라이브러리
content_library = {
    1: {
        "grade_name": "✨MZ세대 프리패스상✨",
        "image_url": "img/grade-1.png",
        "detail_analysis": "미간 분석: 훈계 확률 3% / 입꼬리 분석: '라떼' 시전 가능성 1% / 눈빛 분석: 동료애 지수 27%",
        "total_comment": "당신의 얼굴에선 자유로운 영혼의 기운만 느껴집니다. '꼰대'라는 단어와는 평생 무관할 상입니다. 회식보단 퇴근 후의 워라밸을 즐기세요!"
    },
    2: {
        "grade_name": "🌱예비 꼰대 유망주상🌱",
        "image_url": "img/grade-2.png",
        "detail_analysis": "미간 분석: 훈계 확률 25% / 입꼬리 분석: '라떼' 시전 가능성 30% / 눈빛 분석: 아랫사람 감시 능력 22%",
        "total_comment": "아직은 괜찮습니다. 하지만 미간에 아주 희미한 '불만'의 기운이 서려 있습니다. 방심하는 순간 당신의 입에서 '요즘 애들은...'이 튀어나올 수 있습니다."
    },
    3: {
        "grade_name": "👔팀장급 프로 꼰대상👔",
        "image_url": "img/grade-3.png",
        "detail_analysis": "미간 분석: 훈계 확률 58% / 입꼬리 분석: '라떼' 시전 가능성 65% / 눈빛 분석: 아랫사람 감시 능력 70%",
        "total_comment": "적당히 내려간 입꼬리와 깊어지기 시작한 팔자 주름에서 '라떼는 말이야'의 기운이 느껴집니다. 회식 자리에서 명언 한두 개쯤은 기본으로 장착하셨군요."
    },
    4: {
        "grade_name": "💼부장급 마스터 꼰대상💼",
        "image_url": "img/grade-4.png",
        "detail_analysis": "미간 분석: 훈계 확률 82% / 입꼬리 분석: '라떼' 시전 가능성 91% / 눈빛 분석: 아랫사람 감시 능력 95%",
        "total_comment": "날카로운 눈빛과 굳게 닫힌 입술에서 한 치의 오차도 용납하지 않는 장인의 풍모가 느껴집니다. 당신의 한마디에 분위기는 싸늘해집니다."
    },
    5: {
        "grade_name": "🗿살아있는 화석상🗿",
        "image_url": "img/grade-5.png",
        "detail_analysis": "미간 분석: 훈계 확률 99% / 입꼬리 분석: '라떼' 시전 가능성 100% / 눈빛 분석: CCTV 모드 100%",
        "total_comment": "당신은 꼰대가 아닙니다. 꼰대의 역사를 증명하는 살아있는 화석 그 자체입니다. 박물관에 전시되어 후세에 교훈을 남겨야 할 얼굴입니다. 존경합니다."
    }
}


def get_kkondae_level(score):
    if 0 <= score <= 20:
        return 1
    elif 21 <= score <= 40:
        return 2
    elif 41 <= score <= 60:
        return 3
    elif 61 <= score <= 80:
        return 4
    else:
        return 5

def analyze_face(image):
    """
    얼굴 이미지를 분석하여 꼰대력 점수를 계산하는 함수 (Mediapipe 버전)
    """
    # Mediapipe는 RGB 이미지를 사용
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None, "얼굴을 찾을 수 없어요. 더 정면으로 찍은 사진을 올려보세요!"

    face_landmarks = results.multi_face_landmarks[0]
    
    # 랜드마크 좌표를 numpy 배열로 변환
    h, w, _ = image.shape
    coords = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

    # 1. 입꼬리 각도 계산 (Mediapipe 인덱스 사용)
    left_corner = coords[61]
    right_corner = coords[291]
    mouth_center = coords[13]
    
    angle_left = np.arctan2(left_corner[1] - mouth_center[1], left_corner[0] - mouth_center[0])
    angle_right = np.arctan2(right_corner[1] - mouth_center[1], right_corner[0] - mouth_center[0])
    mouth_angle_deg = np.degrees(angle_left + (np.pi - angle_right)) / 2
    mouth_score = int(max(0, min(40, (mouth_angle_deg - 5) * 10)))

    # 2. 미간 사이 거리 계산
    eyebrow_left = coords[285]
    eyebrow_right = coords[55]
    face_left = coords[234]
    face_right = coords[454]
    face_width = np.linalg.norm(face_left - face_right)
    eyebrow_dist = np.linalg.norm(eyebrow_left - eyebrow_right)
    
    eyebrow_ratio = eyebrow_dist / face_width
    eyebrow_score = int(max(0, min(35, (0.13 - eyebrow_ratio) * 2000)))

    # 3. 눈을 뜬 정도 계산
    def eye_aspect_ratio(eye_coords):
        A = np.linalg.norm(eye_coords[1] - eye_coords[5])
        B = np.linalg.norm(eye_coords[2] - eye_coords[4])
        C = np.linalg.norm(eye_coords[0] - eye_coords[3])
        ear = (A + B) / (2.0 * C)
        return ear

    # 왼쪽 눈(33, 160, 158, 133, 153, 144), 오른쪽 눈(362, 385, 387, 263, 373, 380)
    left_eye = np.array([coords[33], coords[160], coords[158], coords[133], coords[153], coords[144]])
    right_eye = np.array([coords[362], coords[385], coords[387], coords[263], coords[373], coords[380]])
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    eye_score = int(max(0, min(25, (0.30 - avg_ear) * 1000)))

    # 최종 점수 계산 및 랜덤 변수 적용
    deterministic_score = mouth_score + eyebrow_score + eye_score
    variation_range = int(deterministic_score * 0.05)
    variation = random.randint(-variation_range, variation_range)
    total_score = deterministic_score + variation
    total_score = max(0, min(100, total_score))

    return total_score, None



@app.route('/analyze', methods=['POST'])
def analyze():
    # Base64로 인코딩된 이미지 데이터 받기
    image_data_url = request.json.get('image')
    if not image_data_url:
        return jsonify({"error": "이미지 데이터가 없습니다."}), 400

    # 데이터 URL에서 순수 Base64 부분만 추출
    header, encoded = image_data_url.split(",", 1)
    
    # Base64 디코딩하여 이미지로 변환
    image_data = base64.b64decode(encoded)
    image_stream = io.BytesIO(image_data)
    pil_image = Image.open(image_stream)
    
    # OpenCV에서 처리할 수 있는 포맷으로 변경
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    score, error_msg = analyze_face(cv_image)

    if error_msg:
        return jsonify({"error": error_msg}), 400

    level = get_kkondae_level(score)
    result_content = content_library[level]

    response_data = {
        "score": score,
        "grade_name": result_content["grade_name"],
        "image_url": result_content["image_url"],
        "detail_analysis": result_content["detail_analysis"],
        "total_comment": result_content["total_comment"]
    }

    return jsonify(response_data)


# if __name__ == '__main__':
#     app.run(debug=True, port=5001) # 포트 번호를 5001로 변경 (프론트와 겹치지 않게)
