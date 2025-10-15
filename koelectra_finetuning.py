# -*- coding: utf-8 -*-
"""
koelectra_finetuning.py (완전판)
---------------------------------
✅ extracted_data 폴더 Excel로 KcELECTRA 모델 파인튜닝
✅ filtered_label, content 컬럼으로 학습
✅ 라벨 번호와 이름 매핑 저장
✅ 에포크마다 트레인/밸리데이션 어큐러시와 손실 출력
✅ 정확도 향상 위한 하이퍼파라미터 및 옵션 적용
"""

import os
import glob
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("[정보] 사용 장치:", device)


def load_data_from_folder(folder_path="extracted_data"):
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"[오류] '{folder_path}' 폴더에 Excel 파일이 없습니다.")
    print("[정보] 발견된 파일들:", [os.path.basename(f) for f in excel_files])

    all_data = []
    label_num2name = {0: '기사복제', 1: '개인의견', 2: '무의미'}

    for file in excel_files:
        print("[정보] 파일 로딩:", os.path.basename(file))
        df = pd.read_excel(file)
        if 'content' not in df.columns or 'filtered_label' not in df.columns or 'filtered_class' not in df.columns:
            print(f"[경고] {os.path.basename(file)}에 필요한 컬럼 누락")
            continue
        df = df[df['filtered_label'].isin(label_num2name.keys())]
        all_data.append(df[['content', 'filtered_label', 'filtered_class']].dropna())
        print(f"[정보] {os.path.basename(file)}: {len(df)}개 데이터")

    if not all_data:
        raise ValueError("[오류] 처리할 유효한 데이터가 없습니다.")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"[정보] 총 {len(combined_df)}개 데이터 병합 완료")

    print("\n[정보] 라벨별 데이터 분포:")
    label_counts = combined_df['filtered_label'].value_counts()
    for label_num in sorted(label_counts.index):
        name = label_num2name[label_num]
        print(f"  - {label_num}({name}): {label_counts[label_num]}개")
    return combined_df, label_num2name


def prepare_dataset(df, tokenizer, max_length=512):
    df['labels'] = df['filtered_label']
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['content'].tolist(),
        df['labels'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['labels']
    )
    print(f"[정보] 훈련 데이터: {len(train_texts)}개, 검증 데이터: {len(val_texts)}개")

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=max_length
        )

    train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels})
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    return train_dataset, val_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"검증_정확도": acc}


class TrainAccuracyCallback(TrainerCallback):
    def on_train_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get('trainer', None)
        if trainer is None:
            return control
        model = trainer.model
        train_loader = trainer.get_train_dataloader()
        model.eval()
        total, correct = 0, 0
        for batch in train_loader:
            batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        train_acc = correct / total if total > 0 else 0

        # 최근 손실 값 가져오기 (state.log_history 중 가장 마지막 loss)
        train_loss = None
        if state.log_history and 'loss' in state.log_history[-1]:
            train_loss = state.log_history[-1]['loss']

        print(
            f"\n[에포크 {state.epoch:.1f}] 훈련_정확도: {train_acc:.4f} | 훈련_손실: {train_loss:.4f if train_loss is not None else 'N/A'}")
        model.train()
        return control


def main():
    print("[정보] 데이터 로딩 시작...")
    df, label_num2name = load_data_from_folder("extracted_data")

    model_name = "beomi/KcELECTRA-base"
    print(f"\n[정보] KcELECTRA 모델 로딩: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(label_num2name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    print("\n[정보] 데이터셋 전처리 시작...")
    train_dataset, val_dataset = prepare_dataset(df, tokenizer)

    output_dir = "kcelec_model"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=12,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=1000,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="검증_정확도",
        greater_is_better=True,
        report_to="none",
        dataloader_pin_memory=False,
        gradient_accumulation_steps=2,
        fp16=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), TrainAccuracyCallback()]
    )

    print("\n[정보] 모델 훈련 시작...")
    trainer.train()

    print(f"\n[정보] 모델 저장 중: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    import json
    with open(os.path.join(output_dir, "label_info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num2name": label_num2name,
                "name2num": {v: k for k, v in label_num2name.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[완료] 훈련 및 저장 완료.")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  - {key}: {value:.4f}")

    print(f"\n[저장된 파일 목록]")
    print(f"  - 모델: {output_dir}/pytorch_model.bin")
    print(f"  - 토크나이저: {output_dir}/tokenizer.json")
    print(f"  - 라벨 정보: {output_dir}/label_info.json")


if __name__ == "__main__":
    main()
