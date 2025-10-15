# -*- coding: utf-8 -*-
"""
zero_shot_classify_v8_stop1000.py
---------------------------------
✅ 모든 시트 병합 + 3라벨 제로샷 분류
✅ 라벨 1(개인의견)이 1000개 도달 시 자동 중단
✅ 프로젝트 폴더에서 SOCIAL_ 파일 자동 검색 (여러 파일 순차 처리)
✅ 출력 파일명에 _labeled 자동 추가
"""

import argparse
import os
import glob
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# =============================
# 🔹 라벨 정의
# =============================
CANDIDATE_LABELS = [
    "이 문장은 뉴스 기사나 보도문처럼 객관적인 사실을 전달한다. 주관적인 의견이나 감정 표현 없이, 사건, 인물, 혐의, 조치 등의 정보를 중립적으로 기술한다.",
    "이 문장은 개인이 자신의 생각, 판단, 주장, 비판 또는 사회적 행동 참여 의사를 표현한다. 반드시 강한 감정이 드러나지 않아도, 의견이나 해석이 포함되어 있으면 개인 의견으로 본다. 예를 들어 분노, 응원, 청원, 탄원, 지지, 평가 등의 표현이 이에 해당한다.",
    "이 문장은 아동복지법이나 사건의 본질과 관련이 없거나, 단순 언급, 잡담, 회의적 표현 등으로 의미가 불분명하다. 예를 들어 '글쎄', 'ㅋㅋ', '그냥' 등의 표현이 포함된다."
]
HYPOTHESIS_TEMPLATE = "다음 문장은 {}."


# =============================
# 🔹 Excel 파일 자동 검색
# =============================
def find_excel_files(project_dir=None):
    """프로젝트 폴더에서 SOCIAL_ 패턴의 Excel 파일들을 검색"""
    if project_dir is None:
        project_dir = os.getcwd()

    print(f"[INFO] 프로젝트 디렉토리: {project_dir}")

    # 먼저 특정 파일들을 직접 확인
    specific_files = [
        os.path.join(project_dir, "SOCIAL_개인정보보호법_전처리_v4_cleaned.xlsx"),
        os.path.join(project_dir, "SOCIAL_중대재해처벌법_전처리_v4_cleaned.xlsx")
    ]

    excel_files = [f for f in specific_files if os.path.exists(f)]

    if not excel_files:
        # SOCIAL_로 시작하는 xlsx 파일 검색
        pattern = os.path.join(project_dir, "SOCIAL_*.xlsx")
        excel_files = glob.glob(pattern)

        if not excel_files:
            # 다른 패턴도 시도 (한글 파일명 포함)
            all_excel = glob.glob(os.path.join(project_dir, "*.xlsx"))
            social_files = [f for f in all_excel if "SOCIAL" in os.path.basename(f)]
            excel_files = social_files

    print(f"[INFO] 발견된 Excel 파일들: {[os.path.basename(f) for f in excel_files]}")
    return excel_files


# =============================
# 🔹 출력 파일명 생성
# =============================
def generate_output_filename(input_files, base_output=None):
    """입력 파일들을 기반으로 출력 파일명 생성 (_labeled 추가)"""
    if base_output:
        # 사용자가 직접 지정한 경우
        name, ext = os.path.splitext(base_output)
        return f"{name}_labeled{ext}"

    if len(input_files) == 1:
        # 단일 파일인 경우 해당 파일명 기반
        input_path = input_files[0]
        dir_name = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        # _cleaned가 있으면 제거하고 _labeled 추가
        if base_name.endswith("_cleaned"):
            base_name = base_name[:-8]  # "_cleaned" 제거
        return os.path.join(dir_name, f"{base_name}_labeled.xlsx")
    else:
        # 다중 파일인 경우 combined 사용
        dir_name = os.path.dirname(input_files[0])
        return os.path.join(dir_name, "SOCIAL_개인정보보호법_전처리_v4_labeled.xlsx")


# =============================
# 🔹 Zero-shot 분류기 생성
# =============================
def build_classifier(model_name="joeddav/xlm-roberta-large-xnli", device=None):
    import torch
    # CUDA 자동 감지: 없으면 CPU(-1), 있으면 첫 GPU(0) 사용
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    print(f"[INFO] 모델 로드 중: {model_name}, device: {device} (0=GPU, -1=CPU)")
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device
    )


# =============================
# 🔹 문장 분류 함수
# =============================
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


# =============================
# 🔹 단일 파일 처리 함수
# =============================
def process_single_file(input_file, zsc, text_column, target_count):
    """단일 Excel 파일을 처리하고 결과를 반환"""
    print(f"\n[INFO] 파일 처리 중: {os.path.basename(input_file)}")

    # 여러 시트 병합
    xls = pd.ExcelFile(input_file)
    print(f"[INFO] 시트 목록: {xls.sheet_names}")

    df_list = []
    for sheet_name in xls.sheet_names:
        temp_df = pd.read_excel(xls, sheet_name=sheet_name)
        if text_column not in temp_df.columns:
            print(f"[WARN] {sheet_name} 시트에 '{text_column}' 없음 → 건너뜀")
            continue
        temp_df = temp_df[temp_df[text_column].notna()]
        if len(temp_df) == 0:
            print(f"[WARN] {sheet_name} 시트에 유효한 데이터 없음 → 건너뜀")
            continue
        temp_df["source_sheet"] = sheet_name
        temp_df["source_file"] = os.path.basename(input_file)  # 파일 정보 추가
        df_list.append(temp_df)
        print(f"[INFO] {sheet_name} 시트: {len(temp_df)}개 문장")

    if not df_list:
        print(f"[WARN] {os.path.basename(input_file)}에 유효한 시트가 없습니다.")
        return pd.DataFrame(), 0, False

    df = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] {os.path.basename(input_file)}: 총 {len(df)}개 문장")

    # 실시간 분류 + 조기 종료 체크
    results = []
    label1_count = 0
    early_stop = False

    pbar = tqdm(total=len(df), desc=f"분류 중 - {os.path.basename(input_file)}")

    for i, text in enumerate(df[text_column]):
        res = classify_text(zsc, str(text))
        results.append(res)

        if res[3] == 1:  # 라벨 1(개인의견)
            label1_count += 1

        pbar.update(1)

        if label1_count >= target_count:
            print(f"\n🎯 {os.path.basename(input_file)}에서 라벨1(개인의견) {label1_count}개 도달! → 조기 종료")
            df = df.iloc[: i + 1]
            early_stop = True
            break

    pbar.close()

    # 결과 컬럼 추가
    df["p_기사복사"] = [r[0] for r in results]
    df["p_개인의견"] = [r[1] for r in results]
    df["p_무의미"] = [r[2] for r in results]
    df["pred_label"] = [r[3] for r in results]
    df["pred_class"] = df["pred_label"].map({0: "뉴스기사복사", 1: "개인의견", 2: "무의미"})

    print(f"[INFO] {os.path.basename(input_file)} 완료: {len(df)}개 문장 처리 / 개인의견(1): {label1_count}개")

    return df, label1_count, early_stop


# =============================
# 🔹 메인 실행 함수
# =============================
def main():
    parser = argparse.ArgumentParser(description="라벨1(개인의견) 1000개 도달 시 조기 종료 버전 - 다중 파일 지원")
    parser.add_argument("--input", type=str, default=None, help="입력 파일 (.xlsx) - 미지정시 자동 검색")
    parser.add_argument("--project-dir", type=str, default=r"C:\Users\user\PycharmProjects\PythonProject3",
                        help="프로젝트 디렉토리 경로")
    parser.add_argument("--text-column", type=str, default="content", help="문장 텍스트 칼럼명")
    parser.add_argument("--output", type=str, default=None, help="출력 파일명 (미지정시 자동 생성)")
    parser.add_argument("--device", type=int, default=None, help="GPU index (예: 0)")
    parser.add_argument("--target-count", type=int, default=1000, help="라벨 1 목표 개수 (기본 1000)")
    parser.add_argument("--process-all", action="store_true", help="모든 파일을 순차 처리 (기본: False)")
    args = parser.parse_args()

    # ✅ 입력 파일 결정
    if args.input is None:
        # 자동 검색
        excel_files = find_excel_files(args.project_dir)
        if not excel_files:
            raise FileNotFoundError("SOCIAL_*.xlsx 파일을 찾을 수 없습니다.")

        if len(excel_files) == 1:
            input_files = excel_files
            print(f"[INFO] 자동 선택된 파일: {[os.path.basename(f) for f in input_files]}")
        else:
            if args.process_all:
                input_files = excel_files
                print(f"[INFO] 모든 파일 처리 모드: {len(input_files)}개 파일")
            else:
                print("[INFO] 여러 SOCIAL_ 파일이 발견되었습니다:")
                for i, f in enumerate(excel_files):
                    print(f"  {i + 1}. {os.path.basename(f)}")
                print(f"  {len(excel_files) + 1}. 모든 파일 처리")

                choice = input(f"처리할 파일 번호를 선택하세요 (1-{len(excel_files) + 1}): ")
                try:
                    choice_num = int(choice)
                    if choice_num == len(excel_files) + 1:
                        input_files = excel_files
                        print("[INFO] 모든 파일을 순차 처리합니다.")
                    else:
                        input_files = [excel_files[choice_num - 1]]
                except (ValueError, IndexError):
                    print("[ERROR] 잘못된 선택입니다. 첫 번째 파일을 사용합니다.")
                    input_files = [excel_files[0]]
    else:
        # 지정된 파일 사용
        if not os.path.isabs(args.input):
            input_file = os.path.join(args.project_dir, args.input)
        else:
            input_file = args.input

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_file}")
        input_files = [input_file]

    print(f"[INFO] 처리할 파일 수: {len(input_files)}")

    # ✅ 분류기 로드 (한 번만)
    zsc = build_classifier(device=args.device)

    # ✅ 다중 파일 순차 처리
    all_results = []
    total_label1_count = 0

    for input_file in input_files:
        df_result, label1_count, early_stop = process_single_file(
            input_file, zsc, args.text_column, args.target_count - total_label1_count
        )

        if not df_result.empty:
            all_results.append(df_result)
            total_label1_count += label1_count

        # 목표 달성 시 전체 중단
        if total_label1_count >= args.target_count:
            print(f"\n🏁 전체 목표 달성! 총 라벨1(개인의견): {total_label1_count}개")
            break

    if not all_results:
        raise ValueError("처리할 유효한 데이터가 없습니다.")

    # ✅ 모든 결과 합치기
    final_df = pd.concat(all_results, ignore_index=True)

    # ✅ 출력 파일 경로 설정 (_labeled 자동 추가)
    output_file = generate_output_filename(input_files, args.output)

    final_df.to_excel(output_file, index=False)

    print(f"\n✅ 모든 처리 완료 → {output_file}")
    print(f"📊 최종 결과:")
    print(f"  - 처리된 파일 수: {len(all_results)}")
    print(f"  - 총 문장 수: {len(final_df)}")
    print(f"  - 개인의견(라벨1): {total_label1_count}개")

    # 분류 결과 요약
    label_counts = final_df["pred_class"].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(final_df)) * 100
        print(f"  - {label}: {count}개 ({percentage:.1f}%)")


if __name__ == "__main__":
    main()