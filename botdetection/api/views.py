from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb

class BotDetectionView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        file = request.FILES['file']
        df = pd.read_csv(file)

        # Ensure required columns exist
        required_columns = ['Retweet Count', 'Mention Count', 'Follower Count']
        if not all(col in df.columns for col in required_columns):
            return Response({"error": "Missing required columns"}, status=400)

        # Fill missing values with the median for required columns
        df[required_columns] = df[required_columns].fillna(df[required_columns].median())

        # Log Transformation for Skewed Data (Handle cases where values might be zero or negative)
        for col in required_columns:
            df[col] = np.log1p(df[col])

        # Feature Scaling
        scaler = StandardScaler()
        df[required_columns] = scaler.fit_transform(df[required_columns])

        # Train Isolation Forest (Increase Sensitivity)
        iso_forest = IsolationForest(n_estimators=500, contamination=0.05, max_samples='auto', random_state=42)
        df['iso_score'] = iso_forest.fit_predict(df[required_columns])

        # Train One-Class SVM (Increase Sensitivity)
        oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)  # Increased sensitivity
        df['svm_score'] = oc_svm.fit_predict(df[required_columns])

        # Train XGBoost Classifier (If Labels Exist)
        if 'Bot Label' in df.columns:
            y_true = df['Bot Label']
            X_train = df[required_columns]
            y_train = y_true

            xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
            xgb_model.fit(X_train, y_train)
            df['xgb_pred'] = xgb_model.predict(X_train)

            # Final Decision: Combine All Models
            df['final_pred'] = ((df['iso_score'] < 0) | (df['svm_score'] < 0) | (df['xgb_pred'] == 1)).astype(int)

            y_pred = df['final_pred']

            # Metrics Calculation
            precision = round(precision_score(y_true, y_pred, zero_division=0) * 100, 2)
            recall = round(recall_score(y_true, y_pred, zero_division=0) * 100, 2)
            f1 = round(f1_score(y_true, y_pred, zero_division=0) * 100, 2)
            auc_roc = round(roc_auc_score(y_true, y_pred) * 100, 2)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            false_positive_rate = round((fp / (fp + tn) * 100), 2) if (fp + tn) > 0 else 0
            false_negative_rate = round((fn / (fn + tp) * 100), 2) if (fn + tp) > 0 else 0
        else:
            precision, recall, f1, auc_roc = None, None, None, None
            false_positive_rate, false_negative_rate = None, None

        # Prepare Results
        results = []
        for _, row in df.iterrows():
            result = {
                'username': row.get('Username', 'Unknown'),
                'confidence': round(abs(row['iso_score'] * 100), 2),
                'is_bot': row['final_pred'],
            }

            # Add metrics if available
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
            'bot_count': sum(df['final_pred']),
            'genuine_count': len(df) - sum(df['final_pred']),
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
