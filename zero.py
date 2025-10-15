import os
import glob
import pandas as pd
from transformers import pipeline

def zero_shot_update_category(df, text_col='content', category_col='category', classifier=None):
    if classifier is None:
        classifier = pipeline('zero-shot-classification', model="joeddav/xlm-roberta-large-xnli")

    def choose_best_label(row):
        text = str(row[text_col])
        candidates = [c.strip() for c in str(row[category_col]).split(',') if c.strip()]
        if not candidates:
            return None
        result = classifier(text, candidates)
        return result['labels'][0]

    df[category_col] = df.apply(choose_best_label, axis=1)
    return df

def main():
    data_dir = r"C:\Users\user\PycharmProjects\PythonProject3\topic_modeling"
    file_pattern = os.path.join(data_dir, "*.xlsx")
    files = glob.glob(file_pattern)

    if not files:
        print(f"[ERROR] '{data_dir}'에 처리할 Excel 파일이 없습니다.")
        return

    classifier = pipeline('zero-shot-classification', model="joeddav/xlm-roberta-large-xnli")

    for file in files:
        print(f"[INFO] 파일 처리 중: {os.path.basename(file)}")
        df = pd.read_excel(file)

        if 'content' not in df.columns or 'category' not in df.columns:
            print(f"[WARN] {os.path.basename(file)}에 content 또는 category 컬럼이 없습니다. 건너뜀")
            continue

        df_updated = zero_shot_update_category(df, classifier=classifier)
        output_file = os.path.join(data_dir, os.path.splitext(os.path.basename(file))[0] + "_category_updated.xlsx")
        df_updated.to_excel(output_file, index=False)
        print(f"[INFO] 저장 완료: {output_file}")

if __name__ == "__main__":
    main()
