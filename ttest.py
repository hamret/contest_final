# -*- coding: utf-8 -*-
"""
extract_balanced_threshold_1000_v2.py
---------------------------------
✅ 확률 0.8 이상인 문장만 확정 분류
✅ 라벨 0, 1, 2 각각 1000개씩 균등 추출
"""

import pandas as pd

# 파일 경로
input_path = r"/labeled/SOCIAL_개인정보보호법_전처리_v4_labeled.xlsx"
output_path = r"C:\Users\user\PycharmProjects\PythonProject3\labeled08\SOCIAL_개인정보보호법_전처리_v4_labeled_0.8.xlsx"

# 확률 임계값
THRESHOLD = 0.8

# 파일 읽기
df = pd.read_excel(input_path)

# pred_label 대신 확률 기반으로 새 라벨 부여
def apply_threshold(row, threshold=THRESHOLD):
    p0, p1, p2 = row["p_기사복사"], row["p_개인의견"], row["p_무의미"]
    if p0 >= threshold and p0 > p1 and p0 > p2:
        return 0  # 뉴스기사복사
    elif p1 >= threshold and p1 > p0 and p1 > p2:
        return 1  # 개인의견
    elif p2 >= threshold and p2 > p0 and p2 > p1:
        return 2  # 무의미
    else:
        return -1  # 애매함

df["filtered_label"] = df.apply(apply_threshold, axis=1)
label_map = {0: "뉴스기사복사", 1: "개인의견", 2: "무의미", -1: "애매함"}
df["filtered_class"] = df["filtered_label"].map(label_map)

# 애매한 문장 제외
filtered_df = df[df["filtered_label"] != -1]
print(f"[INFO] 필터 후 남은 문장 수: {len(filtered_df)}개")

# 라벨별 분리
df0 = filtered_df[filtered_df["filtered_label"] == 0]
df1 = filtered_df[filtered_df["filtered_label"] == 1]
df2 = filtered_df[filtered_df["filtered_label"] == 2]

print(f"라벨0(뉴스기사복사): {len(df0)}개")
print(f"라벨1(개인의견): {len(df1)}개")
print(f"라벨2(무의미): {len(df2)}개")

# 각 라벨에서 최대 1000개 추출
sample0 = df0.sample(n=min(1000, len(df0)), random_state=42)
sample1 = df1.sample(n=min(1000, len(df1)), random_state=42)
sample2 = df2.sample(n=min(1000, len(df2)), random_state=42)

# 병합 + 셔플
balanced_df = pd.concat([sample0, sample1, sample2], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 저장
balanced_df.to_excel(output_path, index=False)
print(f"\n✅ 0.8 이상 확률 기준으로 균등 추출 완료 → {output_path}")
print(f"총 {len(balanced_df)}개 문장 (라벨별 최대 1000개)")
