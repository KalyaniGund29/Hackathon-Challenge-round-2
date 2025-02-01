from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from .serializers import DetectionResultSerializer

class BotDetectionView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        file = request.FILES['file']
        df = pd.read_csv(file)

        # Ensure required columns exist
        required_columns = ['Retweet Count', 'Mention Count', 'Follower Count']
        for col in required_columns:
            if col not in df.columns:
                return Response({"error": f"Missing required column: {col}"}, status=400)

        # Fill missing values with the median
        df[required_columns] = df[required_columns].fillna(df[required_columns].median())

        # Use Isolation Forest for anomaly detection
        model = IsolationForest(contamination=0.1)
        df['anomaly_score'] = model.fit_predict(df[required_columns])
        df['is_bot'] = df['anomaly_score'] == -1

        # Assuming `bot_label` column represents the ground truth (1 for bot, 0 for genuine)
        if 'Bot Label' in df.columns:
            y_true = df['Bot Label']  # Ground truth labels (1 for bot, 0 for genuine)
            y_pred = df['is_bot'].astype(int)  # Model predictions

            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            auc_roc = roc_auc_score(y_true, y_pred)

            # Confusion Matrix to calculate False Positive and False Negative rates
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            precision, recall, f1, auc_roc = None, None, None, None
            false_positive_rate, false_negative_rate = None, None

        results = []
        for _, row in df.iterrows():
            result = {
                'username': row['Username'],
                'confidence': abs(row['anomaly_score'] * 100),
                'is_bot': row['is_bot'],
            }

            if precision is not None:
                result['metrics'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_roc': auc_roc,
                    'false_positive_rate': false_positive_rate,
                    'false_negative_rate': false_negative_rate
                }
            results.append(result)

        return Response({
            'bot_count': sum(df['is_bot']),
            'genuine_count': len(df) - sum(df['is_bot']),
            'details': results,
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'false_positive_rate': false_positive_rate,
                'false_negative_rate': false_negative_rate
            }
        })
