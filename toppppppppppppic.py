import os
import glob
import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

# labeled 폴더 지정 (읽기/저장 모두)
LABELED_FOLDER = r"C:\Users\user\PycharmProjects\PythonProject3\labeled"

# 불용어 리스트
stopwords = [
    '은', '는', '이', '가', '을', '를', '에', '의', '로', '으로', '와', '과', '도', '만',
    '에서', '부터', '까지', '로부터', '에게', '께', '한테', '보고', '더', '가장', '매우',
    '너무', '정말', '진짜', '아주', '완전', '전혀', '별로', '조금', '좀', '많이', '잘',
    '못', '안', '않', '때문', '위해', '통해', '대해', '관해', '따라', '의해', '에게',
    '그런', '이런', '저런', '어떤', '모든', '각각', '여러', '다른', '같은', '비슷',
    '다양', '특별', '일반', '기본', '주요', '중요', '필요', '가능', '불가능',
    '그냥', '그저', '단지', '오직', '다만', '하지만', '그러나', '그런데', '따라서',
    '그리고', '또한', '또', '역시', '물론', '당연', '확실', '분명', '아마', '혹시'
]

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    metric='cosine',
    random_state=42
)
embedding_model = SentenceTransformer("BM-K/KoSimCSE-roberta-multitask", device="cuda")


def topic_modeling_per_category_in_file(file_path):
    print(f"\n📂 파일 처리 시작: {file_path}")

    df = pd.read_excel(file_path)
    if "category" not in df.columns or "content" not in df.columns:
        print(f"⚠️ 'category' 또는 'content' 컬럼이 없음 → 스킵: {file_path}")
        return

    categories = df["category"].unique()

    for cat in categories:
        cat_df = df[df["category"] == cat]
        texts = cat_df["content"].dropna().tolist()

        split_texts = []
        for text in texts:
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
            split_texts.extend(paragraphs)

        if len(split_texts) < 10:
            print(f"⚠️ {cat}: 데이터가 너무 적어 스킵 ({len(split_texts)}개)")
            continue

        print(f"\n🚀 [{cat}] 토픽모델링 시작 (문단 수: {len(split_texts)})")

        if len(split_texts) < 500:
            min_df_value = 1
        elif len(split_texts) < 2000:
            min_df_value = 2
        else:
            min_df_value = 3

        vectorizer_model = CountVectorizer(
            stop_words=stopwords,
            ngram_range=(1, 2),
            min_df=min_df_value
        )

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            vectorizer_model=vectorizer_model,
            top_n_words=10,
            calculate_probabilities=False,
            verbose=False
        )

        try:
            topics, probs = topic_model.fit_transform(split_texts)
            topic_model.update_topics(split_texts, topics, vectorizer_model=vectorizer_model)

            topic_info = topic_model.get_topic_info()

            safe_cat_name = str(cat).replace("/", "_").replace(" ", "_")
            save_path = os.path.join(LABELED_FOLDER, f"topic_summary_{safe_cat_name}.xlsx")
            topic_info.to_excel(save_path, index=False)

            print(f"✅ [{cat}] 저장 완료 → {save_path}")

        except Exception as e:
            print(f"❌ [{cat}] 처리 중 오류 발생: {e}")


def main():
    excel_files = glob.glob(os.path.join(LABELED_FOLDER, "*.xlsx"))

    if not excel_files:
        print(f"[오류] {LABELED_FOLDER} 폴더에 엑셀 파일이 없습니다.")
        return

    for file in excel_files:
        topic_modeling_per_category_in_file(file)

    print("\n🎯 모든 category별 토픽모델링 완료!")


if __name__ == "__main__":
    main()
