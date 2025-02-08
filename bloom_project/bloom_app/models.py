import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import os

class EmotionAnalyzer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.load_and_train_model()

    def load_and_train_model(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, '2000_сэтгэл_хөдлөл_агуулсан_өгүүлбэр.xlsx')
        df = pd.read_excel(file_path)
        
        self.label_encoder.fit(df['label'])
        df['Label_Encoded'] = self.label_encoder.transform(df['label'])
        
        X = self.vectorizer.fit_transform(df['Questions']).toarray()
        y = df['Label_Encoded']
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict_emotion(self, text):
        text_vectorized = self.vectorizer.transform([text]).toarray()
        prediction = self.model.predict(text_vectorized)
        emotion = self.label_encoder.inverse_transform(prediction)[0]
        confidence = self.model.predict_proba(text_vectorized).max()
        return emotion, confidence