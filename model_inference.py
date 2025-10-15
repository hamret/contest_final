# -*- coding: utf-8 -*-
"""
model_inference.py
---------------------------------
✅ koelec_model 폴더의 학습된 모델 로드
✅ preprocessed 폴더의 전처리된 데이터 라벨링
✅ 예측 결과를 labeled 폴더에 저장
✅ 신뢰도(confidence score) 포함
"""

import os
import glob
import pandas as pd
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from torch.nn.functional import softmax
import warnings

warnings.filterwarnings('ignore')


def load_model_and_tokenizer(model_path="koelec_model"):
    """학습된 모델과 토크나이저 로드"""
    try:
        print(f"[정보] 모델 로드 중: {model_path}")

        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 모델 로드
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # 라벨 정보 로드
        label_info_path = os.path.join(model_path, "label_info.json")
        if os.path.exists(label_info_path):
            with open(label_info_path, "r", encoding="utf-8") as f:
                label_info = json.load(f)
        else:
            # 기본 라벨 정보
            label_info = {
                "num2name": {0: '기사복제', 1: '개인의견', 2: '무의미'},
                "name2num": {'기사복제': 0, '개인의견': 1, '무의미': 2}
            }

        # GPU 사용 가능하면 모델을 GPU로 이동
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        print(f"[정보] 모델 로드 완료 (장치: {device})")
        print(f"[정보] 라벨 정보: {label_info['num2name']}")

        return model, tokenizer, label_info, device

    except Exception as e:
        print(f"[오류] 모델 로드 실패: {e}")
        return None, None, None, None


def predict_text(text, model, tokenizer, device, max_length=512):
    """단일 텍스트에 대한 예측 수행"""
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return None, None

    try:
        # 텍스트 토크나이징
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # 입력을 GPU로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 예측 수행
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # 소프트맥스를 적용해 확률 계산
            probabilities = softmax(logits, dim=-1)

            # 가장 높은 확률의 라벨 선택
            predicted_label = torch.argmax(logits, dim=-1).item()
            confidence_score = probabilities[0][predicted_label].item()

        return predicted_label, confidence_score

    except Exception as e:
        print(f"[경고] 예측 실패 (텍스트: {text[:50]}...): {e}")
        return None, None


def label_data(input_folder="preprocessed", output_folder="labeled", model_path="koelec_model"):
    """데이터 라벨링 메인 함수"""

    # 모델 로드
    model, tokenizer, label_info, device = load_model_and_tokenizer(model_path)
    if model is None:
        return

    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # Excel 파일들 찾기
    excel_files = glob.glob(os.path.join(input_folder, "*.xlsx"))
    if not excel_files:
        print(f"[오류] '{input_folder}' 폴더에 Excel 파일이 없습니다.")
        return

    print(f"[정보] 발견된 파일들: {[os.path.basename(f) for f in excel_files]}")

    for file_path in excel_files:
        print(f"\n[정보] 파일 처리 중: {os.path.basename(file_path)}")

        # 파일 로딩
        try:
            df = pd.read_excel(file_path)
            print(f"[정보] 원본 데이터: {len(df)}행")
        except Exception as e:
            print(f"[오류] 파일 로딩 실패: {e}")
            continue

        # content 컬럼 확인
        if 'content' not in df.columns:
            print(f"[경고] 'content' 컬럼이 없습니다. 건너뜀.")
            continue

        # 예측 수행
        print("[정보] 라벨링 시작...")
        predicted_labels = []
        confidence_scores = []
        predicted_label_names = []

        for idx, text in enumerate(df['content']):
            if idx % 100 == 0:
                print(f"[진행] {idx}/{len(df)} 완료")

            pred_label, confidence = predict_text(text, model, tokenizer, device)

            if pred_label is not None:
                predicted_labels.append(pred_label)
                confidence_scores.append(confidence)
                predicted_label_names.append(label_info['num2name'][str(pred_label)])
            else:
                predicted_labels.append(-1)  # 예측 실패
                confidence_scores.append(0.0)
                predicted_label_names.append('예측실패')

        # 결과를 데이터프레임에 추가
        df['predicted_label'] = predicted_labels
        df['predicted_label_name'] = predicted_label_names
        df['confidence_score'] = confidence_scores

        # 예측 실패한 행 제거 (선택사항)
        df_clean = df[df['predicted_label'] != -1].copy()

        print(f"[정보] 라벨링 완료: {len(df_clean)}행 (예측 실패: {len(df) - len(df_clean)}행)")

        # 라벨별 분포 출력
        if len(df_clean) > 0:
            print("\n[정보] 예측 라벨 분포:")
            label_counts = df_clean['predicted_label_name'].value_counts()
            for label, count in label_counts.items():
                percentage = count / len(df_clean) * 100
                print(f"  - {label}: {count}개 ({percentage:.1f}%)")

            print(f"\n[정보] 평균 신뢰도: {df_clean['confidence_score'].mean():.4f}")

        # 결과 저장
        output_path = os.path.join(output_folder, f"labeled_{os.path.basename(file_path)}")
        try:
            df_clean.to_excel(output_path, index=False)
            print(f"[완료] 저장: {output_path}")
        except Exception as e:
            print(f"[오류] 저장 실패: {e}")


if __name__ == "__main__":
    print("[시작] 데이터 라벨링 시작...")
    label_data()
    print("\n[완료] 모든 파일 라벨링 완료!")
