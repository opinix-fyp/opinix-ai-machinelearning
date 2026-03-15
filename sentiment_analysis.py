import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from rake_nltk import Rake
import nltk
from langdetect import detect
import re
import importlib

# Label mapping
label2id = {'Good': 0, 'Okay': 1, 'Bad': 2, 'Unsure': 3}
id2label = {v: k for k, v in label2id.items()}

# Heuristic seed words used for weak-label generation.
# Kept local so inference imports do not depend on experiment modules.
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


def print_class_distribution_report(df, title='Dataset Class Distribution', underrepresented_threshold=5.0):
    total = len(df)
    print(f"\n{title}:")

    if total == 0:
        print('No rows available for class distribution analysis.')
        return

    counts = df['sentiment_label'].value_counts()
    for class_id in sorted(id2label.keys()):
        class_name = id2label[class_id]
        class_count = int(counts.get(class_name, 0))
        class_pct = (class_count / total) * 100.0
        print(f"{class_name:<7} {class_count:>5} ({class_pct:>5.1f}%)")

    for class_id in sorted(id2label.keys()):
        class_name = id2label[class_id]
        class_count = int(counts.get(class_name, 0))
        class_pct = (class_count / total) * 100.0
        if class_pct < underrepresented_threshold:
            print(
                f"Warning: The '{class_name}' class is severely underrepresented "
                f"({class_pct:.1f}%) and model performance may be unreliable."
            )


# Toggle to run keyword ablation analysis after normal evaluation.
# Set False by default so the core training pipeline runs without experiment overhead.
RUN_KEYWORD_ABLATION = False

# Optional debug traces for weak-label scoring decisions.
WEAK_LABEL_DEBUG = False
WEAK_LABEL_DEBUG_MAX = 20

OKAY_PHRASES = [
    'okay', 'ok', 'fine', 'alright', 'average', 'decent', 'acceptable', 'fair', 'so-so', 'could be better'
]

UNSURE_PHRASES = [
    'not sure', 'unsure', 'maybe', 'hard to say', 'cannot tell', "can't tell", 'no opinion',
    'mixed feelings', 'not certain', 'somewhat', 'kind of'
]

NEGATION_SENSITIVE_POSITIVE = ['good', 'helpful', 'useful', 'great', 'excellent', 'amazing', 'love', 'loved']
NEGATION_SENSITIVE_NEGATIVE = ['bad', 'terrible', 'awful', 'worst', 'poor', 'hate', 'hated', 'disappointing']

# Score margins are tuned to keep labels stable without collapsing most labels to Okay.
GOOD_BAD_MARGIN = 1.0
UNSURE_DOMINANCE_MARGIN = 0.5
MILD_SENTIMENT_GAP = 1.0


class DataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._debug_prints = 0

        self.positive_terms = sorted(set(POSITIVE_KEYWORDS), key=len, reverse=True)
        self.negative_terms = sorted(set(NEGATIVE_KEYWORDS), key=len, reverse=True)
        self.okay_phrases = sorted(set(OKAY_PHRASES), key=len, reverse=True)
        self.unsure_phrases = sorted(set(UNSURE_PHRASES), key=len, reverse=True)

        self.positive_pattern = self._build_phrase_pattern(self.positive_terms)
        self.negative_pattern = self._build_phrase_pattern(self.negative_terms)
        self.okay_pattern = self._build_phrase_pattern(self.okay_phrases)
        self.unsure_pattern = self._build_phrase_pattern(self.unsure_phrases)

        self.negated_positive_pattern = self._build_phrase_pattern(
            [f'not {term}' for term in NEGATION_SENSITIVE_POSITIVE]
        )
        self.negated_negative_pattern = self._build_phrase_pattern(
            [f'not {term}' for term in NEGATION_SENSITIVE_NEGATIVE]
        )

    def _build_phrase_pattern(self, phrases):
        if not phrases:
            return re.compile(r'(?!)')

        escaped = []
        for phrase in phrases:
            escaped_phrase = re.escape(phrase).replace(r'\ ', r'\s+')
            escaped.append(escaped_phrase)

        return re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', flags=re.IGNORECASE)

    def _count_matches(self, pattern, text):
        return len(pattern.findall(text))

    def _normalize_text(self, text):
        return re.sub(r'\s+', ' ', str(text).lower()).strip()

    def infer_sentiment_label(self, text: str) -> str:
        text_norm = self._normalize_text(text)

        scores = {
            'positive': float(self._count_matches(self.positive_pattern, text_norm)),
            'negative': float(self._count_matches(self.negative_pattern, text_norm)),
            'okay': float(self._count_matches(self.okay_pattern, text_norm)),
            'unsure': float(self._count_matches(self.unsure_pattern, text_norm)),
        }

        negated_positive_hits = self._count_matches(self.negated_positive_pattern, text_norm)
        if negated_positive_hits > 0:
            scores['positive'] = max(0.0, scores['positive'] - negated_positive_hits)
            scores['negative'] += negated_positive_hits * 1.5

        negated_negative_hits = self._count_matches(self.negated_negative_pattern, text_norm)
        if negated_negative_hits > 0:
            scores['negative'] = max(0.0, scores['negative'] - negated_negative_hits)
            scores['okay'] += negated_negative_hits * 1.25

        sentiment_gap = scores['positive'] - scores['negative']

        if (
            scores['unsure'] >= 1.0
            and scores['unsure'] >= max(scores['positive'], scores['negative']) + UNSURE_DOMINANCE_MARGIN
        ):
            label = 'Unsure'
        elif scores['positive'] >= 1.0 and scores['negative'] == 0.0:
            label = 'Good'
        elif scores['negative'] >= 1.0 and scores['positive'] == 0.0:
            label = 'Bad'
        elif sentiment_gap >= GOOD_BAD_MARGIN and scores['positive'] >= 1.0:
            label = 'Good'
        elif sentiment_gap <= -GOOD_BAD_MARGIN and scores['negative'] >= 1.0:
            label = 'Bad'
        elif scores['positive'] >= 1.0 and scores['negative'] >= 1.0 and abs(sentiment_gap) <= MILD_SENTIMENT_GAP:
            label = 'Unsure'
        elif scores['okay'] >= 1.0 and abs(sentiment_gap) <= MILD_SENTIMENT_GAP:
            label = 'Okay'
        elif scores['positive'] == 0.0 and scores['negative'] == 0.0 and scores['unsure'] == 0.0:
            label = 'Okay'
        elif abs(sentiment_gap) < GOOD_BAD_MARGIN:
            label = 'Okay'
        else:
            label = 'Good' if sentiment_gap > 0 else 'Bad'

        if WEAK_LABEL_DEBUG and self._debug_prints < WEAK_LABEL_DEBUG_MAX:
            print(
                f"[weak-label-debug] text='{str(text)[:120]}' "
                f"scores={scores} sentiment_gap={sentiment_gap:.2f} label={label}"
            )
            self._debug_prints += 1

        return label

    def load_raw(self):
        return pd.read_csv(self.csv_path).reset_index().rename(columns={'index': 'source_row_id'})

    def melt_dataframe(self, df):
        df = df.copy()

        # Find all feedback columns (starting with q) for melting.
        feedback_cols = [c for c in df.columns if re.match(r'^q\d', c, re.IGNORECASE)]
        if not feedback_cols:
            raise ValueError('No feedback columns found in the dataset. Expected columns like q1, q2, ...')

        id_vars = ['source_row_id']
        if 'dept_metadata' in df.columns:
            id_vars.append('dept_metadata')

        # Melt feedback into a single column for modeling.
        melted = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=feedback_cols,
            var_name='question',
            value_name='feedback_text',
        )
        melted = melted.dropna(subset=['feedback_text'])

        # If labels are already present, use them; otherwise infer a label per feedback text.
        if 'sentiment_label' not in melted.columns:
            melted['sentiment_label'] = melted['feedback_text'].apply(self.infer_sentiment_label)

        melted['label'] = melted['sentiment_label'].map(label2id)
        melted['model_text'] = melted['question'] + ': ' + melted['feedback_text']
        return melted

    def load_and_melt(self):
        # Backward-compatible helper for callers that still use the old API.
        raw_df = self.load_raw()
        return self.melt_dataframe(raw_df)


class Preprocessor:
    def __init__(self, model_name='distilroberta-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def clean_text(self, text):
        # Normalize whitespace and trim text, preserve natural structure for transformers
        text = re.sub(r'\s+', ' ', str(text))
        return text.strip()

    def is_english(self, text):
        try:
            return detect(text) == 'en'
        except Exception:
            return False

    def preprocess(self, df):
        df['model_text'] = df['model_text'].apply(self.clean_text)
        df = df[df['model_text'].apply(self.is_english)]
        return df

    def tokenize_function(self, examples):
        return self.tokenizer(examples['model_text'], padding='max_length', truncation=True, max_length=128)


class SentimentModel:
    def __init__(self, model_name='distilroberta-base', num_labels=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f'Using device: {self.device}')

    def create_torch_dataset(self, hf_dataset, batch_size=8, shuffle=False):
        input_ids = torch.tensor(hf_dataset['input_ids'])
        attention_mask = torch.tensor(hf_dataset['attention_mask'])
        labels = torch.tensor(hf_dataset['label'])
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def train(self, train_dataset, val_dataset, epochs=3, batch_size=8):
        train_loader = self.create_torch_dataset(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = self.create_torch_dataset(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs} completed')
        return val_loader

    def evaluate(self, val_loader):
        # Collect all predictions and labels for a full report and extra metrics.
        preds = []
        labels = []

        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, batch_labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_preds = torch.argmax(outputs.logits, dim=1)
                preds.extend(batch_preds.cpu().tolist())
                labels.extend(batch_labels.cpu().tolist())

        class_ids = list(range(len(label2id)))
        report = classification_report(
            labels,
            preds,
            labels=class_ids,
            target_names=list(label2id.keys()),
            zero_division=0,
        )
        cm = confusion_matrix(labels, preds, labels=class_ids)
        macro_f1 = f1_score(labels, preds, labels=class_ids, average='macro', zero_division=0)
        weighted_f1 = f1_score(labels, preds, labels=class_ids, average='weighted', zero_division=0)

        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
        }

    def save_confusion_matrix_plot(self, confusion_matrix_values, output_path='confusion_matrix.png', show_plot=False):
        try:
            plt = importlib.import_module('matplotlib.pyplot')
            sns = importlib.import_module('seaborn')
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'Confusion matrix plotting requires matplotlib and seaborn. '
                'Install them with: pip install matplotlib seaborn'
            ) from exc

        label_names = list(label2id.keys())
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix_values,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names,
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        if show_plot:
            plt.show()
        plt.close()
        return output_path


class Predictor:
    def __init__(self, model, tokenizer, preprocessor):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor

    def predict(self, text, question=None):
        if question:
            text = question + ': ' + text
        cleaned = self.preprocessor.clean_text(text)
        if not self.preprocessor.is_english(cleaned):
            return 'Unsure'

        inputs = self.tokenizer(cleaned, return_tensors='pt')
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()
        return id2label[pred]


class KeywordExtractor:
    def __init__(self):
        # RAKE uses NLTK stopwords; ensure they are available.
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        self.rake = Rake()

    def extract(self, text, top_n=5):
        self.rake.extract_keywords_from_text(text)
        keywords = self.rake.get_ranked_phrases()[:top_n]
        return keywords


class SummaryGenerator:
    def __init__(self, keyword_extractor, top_n=5):
        self.keyword_extractor = keyword_extractor
        self.top_n = top_n
        self.ordered_labels = list(label2id.keys())
        self.noise_phrases = {
            'dont know',
            "don't know",
            'do not know',
            'not sure',
            'unsure',
            'no opinion',
            'feel strongly either way',
        }

    def _normalize_phrase(self, phrase):
        phrase = re.sub(r'\s+', ' ', str(phrase)).strip().lower()
        phrase = re.sub(r'[^a-z0-9]+', ' ', phrase).strip()
        return phrase

    def _is_noisy_phrase(self, phrase):
        if not phrase:
            return True

        words = phrase.split()
        alpha_chars = sum(1 for char in phrase if char.isalpha())

        if len(words) > 8:
            return True
        if len(words) < 2:
            return True
        if alpha_chars < 3:
            return True
        if phrase in self.noise_phrases:
            return True

        return False

    def _extract_clean_keywords(self, text):
        if not text or not text.strip():
            return []

        # Request a larger candidate pool and then apply deterministic cleanup.
        raw_keywords = self.keyword_extractor.extract(text, top_n=self.top_n * 4)
        cleaned = []
        seen = set()

        for keyword in raw_keywords:
            normalized = self._normalize_phrase(keyword)
            if self._is_noisy_phrase(normalized):
                continue

            duplicate_like = normalized in seen or any(
                normalized in existing or existing in normalized
                for existing in seen
            )
            if duplicate_like:
                continue

            seen.add(normalized)
            cleaned.append(normalized)

            if len(cleaned) >= self.top_n:
                break

        return cleaned

    def _format_keywords(self, keywords):
        if not keywords:
            return 'no clear recurring themes'
        return ', '.join(keywords)

    def _determine_overall_tone(self, percentages):
        good_pct = percentages.get('Good', 0.0)
        okay_pct = percentages.get('Okay', 0.0)
        bad_pct = percentages.get('Bad', 0.0)

        if good_pct >= 50.0 and good_pct >= bad_pct + 10.0:
            return 'mostly positive'
        if bad_pct >= 50.0 and bad_pct >= good_pct + 10.0:
            return 'mostly negative'
        if okay_pct >= 50.0 and good_pct < 40.0 and bad_pct < 40.0:
            return 'mostly neutral'
        return 'mixed'

    def generate(self, prediction_df):
        required_cols = {'feedback_text', 'predicted_label'}
        missing_cols = required_cols.difference(prediction_df.columns)
        if missing_cols:
            missing = ', '.join(sorted(missing_cols))
            raise ValueError(f"Prediction dataframe is missing required columns: {missing}")

        total_responses = len(prediction_df)
        if total_responses == 0:
            return (
                'No responses were available for summary generation, so sentiment distribution and theme extraction '
                'could not be computed.'
            )

        counts = prediction_df['predicted_label'].value_counts().reindex(self.ordered_labels, fill_value=0).astype(int)
        percentages = ((counts / total_responses) * 100.0).to_dict()

        # Keep response boundaries to reduce fragmented phrase extraction.
        good_text = '. '.join(prediction_df.loc[prediction_df['predicted_label'] == 'Good', 'feedback_text'].astype(str))
        bad_text = '. '.join(prediction_df.loc[prediction_df['predicted_label'] == 'Bad', 'feedback_text'].astype(str))

        positive_keywords = self._extract_clean_keywords(good_text)
        negative_keywords = self._extract_clean_keywords(bad_text)

        overall_tone = self._determine_overall_tone(percentages)
        unsure_count = int(counts.get('Unsure', 0))

        distribution_text = ', '.join(
            [
                f"{label}: {int(counts[label])} ({percentages[label]:.1f}%)"
                for label in self.ordered_labels
            ]
        )

        unsure_sentence = (
            f" There are also {unsure_count} unsure responses, indicating some uncertainty or ambiguous feedback."
            if unsure_count > 0
            else ''
        )

        summary = (
            f"Across {total_responses} responses, the sentiment distribution is {distribution_text}, suggesting the overall "
            f"feedback is {overall_tone}. Common positive themes include {self._format_keywords(positive_keywords)}, "
            f"while common negative themes include {self._format_keywords(negative_keywords)}.{unsure_sentence}"
        )
        return summary


def build_prediction_dataframe(df, predictor):
    required_cols = {'question', 'feedback_text'}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        missing = ', '.join(sorted(missing_cols))
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    predictor.model.eval()
    predictions = [
        predictor.predict(text=row.feedback_text, question=row.question)
        for row in df[['question', 'feedback_text']].itertuples(index=False)
    ]

    return pd.DataFrame(
        {
            'question': df['question'].astype(str).values,
            'feedback_text': df['feedback_text'].astype(str).values,
            'predicted_label': predictions,
        }
    )


if __name__ == '__main__':
    print('Starting sentiment_analysis.py')

    # Load the provided sample dataset.
    csv_path = os.path.join(os.path.dirname(__file__), 'opinix_sample_dataset.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected dataset at {csv_path}")

    loader = DataLoader(csv_path)
    raw_df = loader.load_raw()

    # Split raw rows first to prevent leakage from shared source survey rows.
    train_raw_df, val_raw_df = train_test_split(raw_df, test_size=0.2, random_state=42)

    # Sanity check: train/validation should not share original survey rows.
    train_source_ids = set(train_raw_df['source_row_id'].unique())
    val_source_ids = set(val_raw_df['source_row_id'].unique())
    overlap_count = len(train_source_ids.intersection(val_source_ids))
    print(f"Source row ID overlap between train and validation: {overlap_count}")
    if overlap_count == 0:
        print('Sanity check passed: no source_row_id overlap detected.')
    else:
        print('Warning: source_row_id overlap detected. This may indicate data leakage.')

    train_df = loader.melt_dataframe(train_raw_df)
    val_df = loader.melt_dataframe(val_raw_df)

    preprocessor = Preprocessor()
    train_df = preprocessor.preprocess(train_df)
    val_df = preprocessor.preprocess(val_df)

    print_class_distribution_report(train_df, title='Dataset Class Distribution (Training)')
    print_class_distribution_report(val_df, title='Dataset Class Distribution (Validation)')

    # Convert to Dataset
    train_dataset = Dataset.from_pandas(train_df[['model_text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['model_text', 'label']])

    # Tokenize
    train_dataset = train_dataset.map(preprocessor.tokenize_function, batched=True)
    val_dataset = val_dataset.map(preprocessor.tokenize_function, batched=True)

    # Model
    model = SentimentModel()

    # Create PyTorch datasets for training and evaluation
    val_loader = model.train(train_dataset, val_dataset, epochs=3, batch_size=8)

    # Save the trained model and tokenizer
    model.model.save_pretrained('./saved_model')
    model.tokenizer.save_pretrained('./saved_model')

    predictor = Predictor(model.model, model.tokenizer, preprocessor)

    prediction_df = build_prediction_dataframe(val_df, predictor)
    summary_generator = SummaryGenerator(KeywordExtractor())
    summary_text = summary_generator.generate(prediction_df)

    print('Generated Summary:')
    print(summary_text)

    summary_output_path = os.path.join('saved_model', 'summary.txt')
    with open(summary_output_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write(summary_text + '\n')
    print(f"Summary saved to: {summary_output_path}")

    metrics = model.evaluate(val_loader)

    print('Classification Report:')
    print(metrics['classification_report'])
    print('Confusion Matrix (rows=true labels, cols=predicted labels):')
    print(pd.DataFrame(
        metrics['confusion_matrix'],
        index=[f"true_{label}" for label in label2id.keys()],
        columns=[f"pred_{label}" for label in label2id.keys()],
    ))
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}")

    cm_plot_path = model.save_confusion_matrix_plot(
        metrics['confusion_matrix'],
        output_path=os.path.join('saved_model', 'confusion_matrix.png'),
        show_plot=True,
    )
    print(f"Confusion matrix plot saved to: {cm_plot_path}")

    # ----- Optional Experiment Section: Keyword Ablation (begin) -----
    if RUN_KEYWORD_ABLATION:
        from keyword_ablation import run_keyword_ablation_experiment

        val_eval_df = val_df[['model_text', 'sentiment_label', 'label']].copy()
        run_keyword_ablation_experiment(
            predictor=predictor,
            val_eval_df=val_eval_df,
            class_names=list(label2id.keys()),
        )
    # ----- Optional Experiment Section: Keyword Ablation (end) -----

