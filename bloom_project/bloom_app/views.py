import os
import re
import string
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .serializers import QuestionSerializer, PredictionResponseSerializer
from .models import EmotionAnalyzer
from .serializers import EmotionSerializer, EmotionResponseSerializer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# === Load Mongolian Stopwords ===
APP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(APP_DIR, 'stopwords_mn.txt'), 'r', encoding='utf-8') as f:
    stopwords_mn = set(word.strip() for word in f.readlines())


# === Text Cleaning Function ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\u0400-\u04FF\u1800-\u18AF\u1200-\u137F\u2D30-\u2D7F\u4E00-\u9FFFa-zA-Z\s]', '', text)
    words = text.split()
    cleaned = [word for word in words if word not in stopwords_mn]
    return ' '.join(cleaned)

# === Load Dataset ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, '8000_Блоом_түвшин_датасэт.xlsx')
df = pd.read_excel(file_path)
df.dropna(subset=['Questions', 'Category'], inplace=True)
df = df[df['Questions'].str.len() > 10]
df['Cleaned_Questions'] = df['Questions'].apply(clean_text)

# === Encode Labels ===
label_encoder = LabelEncoder()
df['Label_Encoded'] = label_encoder.fit_transform(df['Category'])

# === TF-IDF Vectorizer ===
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Cleaned_Questions']).toarray()
y = df['Label_Encoded']

# === Train Model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
for _ in tqdm(range(1), desc="Training Progress", total=1):
    model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# === Prediction Function ===
def predict_bloom_level(question):
    cleaned = clean_text(question)
    vec = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vec)
    return label_encoder.inverse_transform(prediction)[0]

# === Index View (HTML Form) ===
def index(request):
    if request.method == 'POST':
        question = request.POST['question']
        bloom_level = predict_bloom_level(question)
        return render(request, 'bloom_app/index.html', {'bloom_level': bloom_level})
    return render(request, 'bloom_app/index.html')

# === API Endpoint ===
@api_view(['POST'])
def predict_bloom_level_api(request):
    try:
        serializer = QuestionSerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            cleaned = clean_text(question)
            vec = vectorizer.transform([cleaned]).toarray()
            prediction = model.predict(vec)
            confidence = model.predict_proba(vec).max()

            response_data = {
                'bloom_level': label_encoder.inverse_transform(prediction)[0],
                'confidence': round(float(confidence), 3)
            }

            response_serializer = PredictionResponseSerializer(data=response_data)
            if response_serializer.is_valid():
                return Response(response_serializer.data, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

from .models import EmotionAnalyzer  # Update import

# Initialize emotion analyzer
emotion_analyzer = EmotionAnalyzer()

@api_view(['POST'])
def predict_emotion_api(request):
    try:
        # Validate input data
        serializer = EmotionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {'error': 'Invalid input data', 'details': serializer.errors}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # Get text from validated data
        text = serializer.validated_data['text']
        
        # Predict emotion
        emotion, confidence = emotion_analyzer.predict_emotion(text)
        
        # Prepare response
        response_data = {
            'emotion': emotion,
            'confidence': float(confidence)
        }
        
        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# Test with curl:
# curl -X POST http://localhost:8000/api/emotion/ -H "Content-Type: application/json" -d "{\"text\":\"I am feeling happy today\"}"

@api_view(['POST'])
def predict_emotion(request):
    try:
        serializer = EmotionSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            emotion, confidence = emotion_analyzer.predict_emotion(text)
            
            response_data = {
                'emotion': emotion,
                'confidence': confidence
            }
            
            response_serializer = EmotionResponseSerializer(data=response_data)
            if response_serializer.is_valid():
                return Response(response_serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def export_emotion_excel(request):
    try:
        serializer = EmotionSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            emotion, confidence = emotion_analyzer.predict_emotion(text)
            
            df = pd.DataFrame({
                'Text': [text],
                'Emotion': [emotion],
                'Confidence': [confidence]
            })
            
            excel_file = io.BytesIO()
            df.to_excel(excel_file, index=False)
            excel_file.seek(0)
            
            response = HttpResponse(
                excel_file.read(),
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            response['Content-Disposition'] = 'attachment; filename=emotion_analysis.xlsx'
            return response
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)