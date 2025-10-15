# -*- coding: utf-8 -*-
"""
data_preprocessing.py
---------------------------------
✅ preprocessing 폴더 내 Excel 파일 전처리
✅ content 컬럼 기준으로 다음 조건 적용:
  - 똑같은 단어가 15개 이상 반복되는 행 삭제
  - '_x00D_' 문자열 제거
  - 이모티콘 제거
  - '❌회원간의 거래분쟁에 대한 공론화 금지❌' 문자열 포함 행 삭제
  - 15글자 이상 중복되는 글의 행 삭제
✅ 전처리된 파일을 preprocessed 폴더에 저장
"""

import os
import glob
import pandas as pd
import re
import emoji
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


def clean_text(text):
    """텍스트 전처리 함수"""
    if pd.isna(text) or not isinstance(text, str):
        return text

    # _x00D_ 문자열 제거
    text = text.replace('_x00D_', '')

    # 이모티콘 제거
    text = emoji.replace_emoji(text, replace='')

    return text.strip()


def has_repeated_words(text, threshold=15):
    """똑같은 단어가 threshold개 이상 반복되는지 확인"""
    if pd.isna(text) or not isinstance(text, str):
        return False

    # 공백으로 단어 분리
    words = text.split()
    if len(words) < threshold:
        return False

    # 단어 빈도 계산
    word_counts = Counter(words)

    # threshold 이상 반복되는 단어가 있는지 확인
    for word, count in word_counts.items():
        if count >= threshold:
            return True
    return False


def has_forbidden_string(text):
    """금지된 문자열이 포함되어 있는지 확인"""
    if pd.isna(text) or not isinstance(text, str):
        return False

    forbidden_strings = [
        '❌회원간의 거래분쟁에 대한 공론화 금지❌',
        '회원간의 거래분쟁에 대한 공론화 금지'
    ]

    for forbidden in forbidden_strings:
        if forbidden in text:
            return True
    return False


def find_duplicate_substrings(text, min_length=15):
    """15글자 이상 중복되는 부분 문자열이 있는지 확인"""
    if pd.isna(text) or not isinstance(text, str) or len(text) < min_length * 2:
        return False

    # 공백 제거한 텍스트로 중복 확인
    clean_text_for_dup = re.sub(r'\s+', '', text)

    for i in range(len(clean_text_for_dup) - min_length + 1):
        substring = clean_text_for_dup[i:i + min_length]
        # 해당 부분 문자열이 텍스트의 다른 부분에도 나타나는지 확인
        if clean_text_for_dup.count(substring) > 1:
            return True
    return False


def preprocess_data(input_folder="preprocessing", output_folder="preprocessed"):
    """전처리 메인 함수"""

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

        original_count = len(df)

        # 2. 금지된 문자열 포함 행 삭제
        before_forbidden = len(df)
        df = df[~df['content'].apply(has_forbidden_string)]
        after_forbidden = len(df)
        print(f"[정보] 금지 문자열 제거: {before_forbidden} -> {after_forbidden} ({before_forbidden - after_forbidden}행 삭제)")

        # 3. 15글자 이상 중복되는 글의 행 삭제
        before_duplicate = len(df)
        df = df[~df['content'].apply(find_duplicate_substrings)]
        after_duplicate = len(df)
        print(f"[정보] 중복 부분문자열 제거: {before_duplicate} -> {after_duplicate} ({before_duplicate - after_duplicate}행 삭제)")

        # 4. 텍스트 정제 (문자열 제거, 이모티콘 제거)
        df['content'] = df['content'].apply(clean_text)

        # 5. 빈 텍스트나 너무 짧은 텍스트 제거
        before_empty = len(df)
        df = df[(df['content'].str.len() > 5) & (df['content'].notna())]
        after_empty = len(df)
        print(f"[정보] 빈/짧은 텍스트 제거: {before_empty} -> {after_empty} ({before_empty - after_empty}행 삭제)")

        print(
            f"[결과] 총 {original_count} -> {len(df)}행 ({original_count - len(df)}행 삭제, {len(df) / original_count * 100:.1f}% 남음)")

        # 전처리된 파일 저장
        output_path = os.path.join(output_folder, f"cleaned_{os.path.basename(file_path)}")
        try:
            df.to_excel(output_path, index=False)
            print(f"[완료] 저장: {output_path}")
        except Exception as e:
            print(f"[오류] 저장 실패: {e}")


if __name__ == "__main__":
    print("[시작] 데이터 전처리 시작...")
    preprocess_data()
    print("\n[완료] 모든 파일 전처리 완료!")
