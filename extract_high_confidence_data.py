# -*- coding: utf-8 -*-
"""
extract_high_confidence_data.py
---------------------------------
✅ work 폴더 내 Excel 파일에서 각 라벨별 확률 0.7 이상 데이터 추출
✅ 각 라벨당 최대 1000개씩 저장
✅ 1000개 미만이면 해당 라벨은 수집 중지
"""

import os
import glob
import pandas as pd


def extract_high_confidence_data():
    # work 폴더 경로
    work_dir = "work"

    if not os.path.exists(work_dir):
        print(f"[ERROR] '{work_dir}' 폴더를 찾을 수 없습니다.")
        return

    # Excel 파일 검색
    excel_files = glob.glob(os.path.join(work_dir, "*.xlsx"))
    if not excel_files:
        print(f"[ERROR] '{work_dir}' 폴더에 Excel 파일이 없습니다.")
        return

    print(f"[INFO] 발견된 파일들: {[os.path.basename(f) for f in excel_files]}")

    # 모든 데이터 병합
    all_data = []
    for file_path in excel_files:
        try:
            print(f"[INFO] 파일 로딩: {os.path.basename(file_path)}")
            df = pd.read_excel(file_path)

            # 필요한 컬럼 확인
            required_cols = ['p_기사복사', 'p_개인의견', 'p_무의미']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"[WARN] {os.path.basename(file_path)}에 다음 컬럼이 없습니다: {missing_cols}")
                continue

            all_data.append(df)
            print(f"[INFO] {os.path.basename(file_path)}: {len(df)}개 행 로딩")

        except Exception as e:
            print(f"[ERROR] {os.path.basename(file_path)} 처리 중 오류: {e}")

    if not all_data:
        print("[ERROR] 처리할 유효한 데이터가 없습니다.")
        return

    # 모든 데이터 병합
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"[INFO] 총 {len(combined_df)}개 행 병합 완료")

    # 각 라벨별로 0.7 이상인 데이터 추출
    labels = [
        ('p_기사복사', '기사복사'),
        ('p_개인의견', '개인의견'),
        ('p_무의미', '무의미')
    ]

    extracted_data = {}

    for prob_col, label_name in labels:
        print(f"\n[INFO] {label_name} 라벨 처리 중...")

        # 해당 라벨 확률이 0.7 이상인 행 필터링
        high_conf_data = combined_df[combined_df[prob_col] >= 0.7].copy()

        print(f"[INFO] {label_name}: 확률 0.7 이상인 데이터 {len(high_conf_data)}개 발견")

        if len(high_conf_data) == 0:
            print(f"[WARN] {label_name}: 조건에 맞는 데이터가 없어 수집 중지")
            continue
        elif len(high_conf_data) < 1000:
            print(f"[WARN] {label_name}: 데이터가 {len(high_conf_data)}개로 1000개 미만이지만 모두 수집")
            extracted_data[label_name] = high_conf_data
        else:
            # 확률 순으로 정렬 후 상위 1000개 선택
            high_conf_data = high_conf_data.sort_values(by=prob_col, ascending=False)
            extracted_data[label_name] = high_conf_data.head(1000)
            print(f"[INFO] {label_name}: 상위 1000개 데이터 선택")

    # 각 라벨별 데이터 저장
    output_dir = "extracted_data"
    os.makedirs(output_dir, exist_ok=True)

    for label_name, data in extracted_data.items():
        output_file = os.path.join(output_dir, f"{label_name}_high_confidence.xlsx")
        data.to_excel(output_file, index=False)
        print(f"[INFO] {label_name}: {len(data)}개 데이터를 {output_file}에 저장")

    # 요약 정보 출력
    print(f"\n{'=' * 50}")
    print("추출 결과 요약")
    print(f"{'=' * 50}")

    total_extracted = 0
    for label_name, data in extracted_data.items():
        print(f"{label_name}: {len(data)}개")
        total_extracted += len(data)

    print(f"총 추출된 데이터: {total_extracted}개")
    print(f"저장 위치: {output_dir} 폴더")

    print(f"\n✅ 데이터 추출 완료!")


def main():
    extract_high_confidence_data()


if __name__ == "__main__":
    main()
