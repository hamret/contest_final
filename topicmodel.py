import os
import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

# 경로 설정
DATA_DIR = r"C:\Users\user\PycharmProjects\PythonProject3\work"  # 본인 데이터셋 폴더로 변경
SAVE_DIR = os.path.join(DATA_DIR, "topic_results")
os.makedirs(SAVE_DIR, exist_ok=True)

# 한국어 불용어 리스트 (기존에 있던 리스트 활용하거나 필요시 수정)
stopwords = [
    '기자','보도','관련','관계자','위해','통해','이번','것','등',
    '제정','시행','발표','밝혔다','전했다','논의','위한',
    '윤석열','이재명','김건희','국민의힘','더불어민주당','정당','대통령','총리'
]

# UMAP 모델 설정
umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)

# 문장 임베딩 모델 (KoSentCSE 로버타 멀티태스크)
embedding_model = SentenceTransformer("BM-K/KoSimCSE-roberta-multitask")

# 해당 폴더 내 모든 Excel 파일 처리
excel_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".xlsx")]

for file_path in excel_files:
    print(f"\n📂 파일 처리: {os.path.basename(file_path)}")
    df = pd.read_excel(file_path)

    # 확인할 컬럼명, 필요시 수정
    if "category" not in df.columns or "content" not in df.columns:
        print(f"⚠️ category 또는 content 컬럼 누락: {file_path}")
        continue

    categories = df["category"].dropna().unique()

    for cat in categories:
        cat_df = df[df["category"] == cat]
        texts = cat_df["content"].dropna().tolist()

        # 문단 단위 분할, 30자 이상만
        split_texts = []
        for text in texts:
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
            split_texts.extend(paragraphs)

        if len(split_texts) < 10:
            print(f"⚠️ {cat} 데이터가 너무 적어 스킵 ({len(split_texts)}개)")
            continue

        print(f"\n🚀 {cat} 토픽 모델링 시작 (문단 수: {len(split_texts)})")

        # min_df 자동 조정
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

        # BERTopic 모델 초기화
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
            save_path = os.path.join(SAVE_DIR, f"topic_summary_{cat}.xlsx")
            topic_info.to_excel(save_path, index=False)
            print(f"✅ 저장 완료: {save_path}")

        except Exception as e:
            print(f"❌ {cat} 처리 중 오류: {e}")

print("\n🎉 모든 파일 처리 완료!")
