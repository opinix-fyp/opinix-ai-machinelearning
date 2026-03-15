import re
from sklearn.metrics import classification_report, f1_score

# Reuse the exact same heuristic seed words used for weak-label generation.
POSITIVE_KEYWORDS = [
    'good', 'great', 'amazing', 'fantastic', 'excellent', 'loved', 'awesome',
    'best', 'positive', 'enjoyed', 'well', 'incredible', 'happy', 'top', 'bravo',
    'recommend', 'love', 'engaging', 'informative', 'useful', 'helpful', 'seamless',
    'wonderful', 'smooth', 'best'
]

NEGATIVE_KEYWORDS = [
    'bad', 'terrible', 'worst', 'awful', 'disappointing', 'hated', 'poor',
    'frustrating', 'never', 'waste', 'boring', 'confusing', 'disorganized',
    'cold', 'not', 'no', 'problem', 'slow', 'late', 'ruined', 'poorly', 'hate'
]

HEURISTIC_KEYWORDS = sorted(set(POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS), key=len, reverse=True)


def build_heuristic_keyword_pattern(keywords=None):
    terms = HEURISTIC_KEYWORDS if keywords is None else list(keywords)
    if not terms:
        return re.compile(r'(?!)')

    escaped_terms = [re.escape(term) for term in terms]
    pattern = r'\b(?:' + '|'.join(escaped_terms) + r')\b'
    return re.compile(pattern, flags=re.IGNORECASE)


def mask_heuristic_keywords(text, keyword_pattern, placeholder='[MASKWORD]'):
    return keyword_pattern.sub(placeholder, str(text))


def contains_heuristic_keyword(text, keyword_pattern):
    return keyword_pattern.search(str(text)) is not None


def _print_eval_block(y_true, y_pred, class_names, title):
    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, labels=class_names, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=class_names, average='weighted', zero_division=0)

    print(f"\n=== {title} ===")
    print(report)
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")

    return {
        'classification_report': report,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
    }


def _predict_texts(predictor, texts):
    return [predictor.predict(text) for text in texts]


def run_keyword_ablation_experiment(predictor, val_eval_df, class_names):
    required_columns = {'model_text', 'sentiment_label'}
    missing_columns = required_columns.difference(val_eval_df.columns)
    if missing_columns:
        raise ValueError(
            f"Validation dataframe missing required columns: {sorted(missing_columns)}"
        )

    keyword_pattern = build_heuristic_keyword_pattern()

    original_texts = val_eval_df['model_text'].astype(str).tolist()
    y_true = val_eval_df['sentiment_label'].astype(str).tolist()

    original_preds = _predict_texts(predictor, original_texts)
    original_metrics = _print_eval_block(
        y_true,
        original_preds,
        class_names,
        title='Original Validation Performance',
    )

    masked_texts = [
        mask_heuristic_keywords(text, keyword_pattern, placeholder='[MASKWORD]')
        for text in original_texts
    ]
    masked_preds = _predict_texts(predictor, masked_texts)
    masked_metrics = _print_eval_block(
        y_true,
        masked_preds,
        class_names,
        title='Masked-Keyword Validation Performance',
    )

    keyword_present_mask = val_eval_df['model_text'].apply(
        lambda text: contains_heuristic_keyword(text, keyword_pattern)
    )

    keyword_present_df = val_eval_df[keyword_present_mask].copy()
    keyword_absent_df = val_eval_df[~keyword_present_mask].copy()

    print("\n=== Keyword Subset Sizes ===")
    print(f"keyword_present rows: {len(keyword_present_df)}")
    print(f"keyword_absent rows: {len(keyword_absent_df)}")

    if len(keyword_present_df) > 0:
        present_texts = keyword_present_df['model_text'].astype(str).tolist()
        present_true = keyword_present_df['sentiment_label'].astype(str).tolist()
        present_preds = _predict_texts(predictor, present_texts)
        keyword_present_metrics = _print_eval_block(
            present_true,
            present_preds,
            class_names,
            title='Keyword-Present Subset Performance',
        )
    else:
        keyword_present_metrics = None
        print("\n=== Keyword-Present Subset Performance ===")
        print("No rows available in keyword_present subset.")

    if len(keyword_absent_df) > 0:
        absent_texts = keyword_absent_df['model_text'].astype(str).tolist()
        absent_true = keyword_absent_df['sentiment_label'].astype(str).tolist()
        absent_preds = _predict_texts(predictor, absent_texts)
        keyword_absent_metrics = _print_eval_block(
            absent_true,
            absent_preds,
            class_names,
            title='Keyword-Absent Subset Performance',
        )
    else:
        keyword_absent_metrics = None
        print("\n=== Keyword-Absent Subset Performance ===")
        print("No rows available in keyword_absent subset.")

    return {
        'original_validation': original_metrics,
        'masked_validation': masked_metrics,
        'keyword_present': keyword_present_metrics,
        'keyword_absent': keyword_absent_metrics,
    }
