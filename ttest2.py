# -*- coding: utf-8 -*-
"""
zero_shot_classify_v8_threshold_save.py
---------------------------------
✅ data_clean 폴더 내 모든 SOCIAL_*.xlsx 파일 처리
✅ 각 라벨별 확률 0.7 이상인 문장만 최대 1000개씩 저장
✅ labeled 폴더에 각 라벨별 파일 저장
"""

import os
import glob
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# 라벨 정의 및 템플릿 (생략 가능하면 기존과 동일)
CANDIDATE_LABELS = [
    "이 문장은 뉴스 기사나 보도문처럼 객관적인 사실을 전달합니다. 주관적인 의견이나 감정 표현 없이, 사건, 인물, 혐의, 조치 등의 정보를 중립적으로 기술해야 합니다. 아래 예시는 이해를 돕기 위한 참고용일 뿐이며, 단순히 예시만 보고 판별하지 말아야 합니다. 예시: \"정부는 2025년 예산안을 확정했다.\", \"경찰은 이번 사건에 대해 수사를 진행 중이다.\", \"신규 코로나19 확진자 수가 1000명을 기록했다.\"",
    "이 문장은 개인이 자신의 생각, 판단, 주장, 비판 또는 사회적 행동 참여 의사를 표현합니다. 반드시 강한 감정이 드러나지 않아도, 의견이나 해석이 포함되어 있으면 개인 의견으로 봅니다. 예시에만 근거하지 말고 문장의 의미와 의도를 고려해 분류해야 합니다. 예시: \"나는 이번 정책이 매우 위험하다고 생각한다.\", \"이 법안에 반대하는 목소리가 커지고 있다.\", \"성장률 전망에 대해 희망적이다.\"",
    "이 문장은 아동복지법이나 사건의 본질과 관련이 없거나, 단순 언급, 잡담, 회의적 표현 등으로 의미가 불분명합니다. 예시: \"글쎄, 잘 모르겠네요.\", \"ㅋㅋ 오늘은 피곤하다.\", \"별 의미 없는 댓글입니다.\" 예시에만 의존하지 말고 맥락을 분석해 판단하세요."
]

HYPOTHESIS_TEMPLATE = "다음 문장은 {}."


def build_classifier(model_name="joeddav/xlm-roberta-large-xnli", device=None):
    import torch
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    print(f"[INFO] 모델 로드 중: {model_name}, device: {device} (0=GPU, -1=CPU)")
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device
    )


def classify_text(zsc, text):
    if not isinstance(text, str) or text.strip() == "":
        return (0.0, 0.0, 0.0, 2)
    res = zsc(
        text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    scores = [dict(zip(res["labels"], res["scores"])).get(label, 0.0) for label in CANDIDATE_LABELS]
    pred_label = int(scores.index(max(scores)))
    return (*scores, pred_label)


def process_file(input_file, zsc, text_column="content"):
    print(f"\n[INFO] 파일 처리 중: {os.path.basename(input_file)}")

    xls = pd.ExcelFile(input_file)
    print(f"[INFO] 시트 목록: {xls.sheet_names}")

    df_list = []
    for sheet_name in xls.sheet_names:
        temp_df = pd.read_excel(xls, sheet_name=sheet_name)
        if text_column not in temp_df.columns:
            print(f"[WARN] {sheet_name} 시트에 '{text_column}' 칼럼 없음 → 건너뜀")
            continue
        temp_df = temp_df[temp_df[text_column].notna()]
        if len(temp_df) == 0:
            print(f"[WARN] {sheet_name} 시트에 유효한 데이터 없음 → 건너뜀")
            continue
        temp_df["source_sheet"] = sheet_name
        temp_df["source_file"] = os.path.basename(input_file)
        df_list.append(temp_df)
        print(f"[INFO] {sheet_name} 시트: {len(temp_df)}개 문장")

    if not df_list:
        print(f"[WARN] {os.path.basename(input_file)}에 유효한 시트가 없습니다.")
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] {os.path.basename(input_file)}: 총 {len(df)}개 문장 병합 완료")

    df = df.drop_duplicates(subset=[text_column])
    print(f"[INFO] {os.path.basename(input_file)}: 중복 제거 후 {len(df)}개 문장")

    # 분류
    tqdm.pandas(desc=f"분류 중 - {os.path.basename(input_file)}")
    results = df[text_column].progress_apply(lambda x: classify_text(zsc, str(x)))

    df["p_기사복사"] = results.apply(lambda t: t[0])
    df["p_개인의견"] = results.apply(lambda t: t[1])
    df["p_무의미"] = results.apply(lambda t: t[2])
    df["pred_label"] = results.apply(lambda t: t[3])
    df["pred_class"] = df["pred_label"].map({0: "뉴스기사복사", 1: "개인의견", 2: "무의미"})

    return df


def main():
    data_dir = "data"
    labeled_dir = "labeled"
    os.makedirs(labeled_dir, exist_ok=True)

    # 분류기 생성
    zsc = build_classifier()

    # data_clean 폴더 내 모든 SOCIAL_*.xlsx 파일 검색
    input_files = glob.glob(os.path.join(data_dir, "SOCIAL_*.xlsx"))
    print(f"[INFO] 처리할 파일들: {[os.path.basename(f) for f in input_files]}")

    # 모든 파일 결과 합치기
    all_df = pd.DataFrame()

    for file in input_files:
        df = process_file(file, zsc)

        if df.empty:
            continue

        all_df = pd.concat([all_df, df], ignore_index=True)

    # 각 라벨별로 확률 0.7 이상인 문장만 필터링 후 최대 1000개 추출 및 저장
    for i, label_name in enumerate(["기사복사", "개인의견", "무의미"]):
        filtered = all_df[all_df[f"p_{label_name}"] >= 0.7]
        filtered = filtered.head(1000)  # 최대 1000개

        output_file = os.path.join(labeled_dir, f"{label_name}_filtered_labeled.xlsx")
        filtered.to_excel(output_file, index=False)
        print(f"[INFO] {label_name} 라벨의 {len(filtered)}개 문장을 {output_file}에 저장했습니다.")


if __name__ == "__main__":
    main()
