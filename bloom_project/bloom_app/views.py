import os
import pandas as pd
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import QuestionSerializer, PredictionResponseSerializer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from .models import EmotionAnalyzer
from .serializers import EmotionSerializer, EmotionResponseSerializer
import pandas as pd
import io

# Load and preprocess data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, '8000_Блоом_түвшин_датасэт.xlsx')
df = pd.read_excel(file_path)
label_encoder = LabelEncoder()
df['Label_Encoded'] = label_encoder.fit_transform(df['Category'])
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['Questions']).toarray()
y = df['Label_Encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
for _ in tqdm(range(1), desc="Training Progress", total=1):
    model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

def predict_bloom_level(question):
    new_question_vectorized = vectorizer.transform([question]).toarray()
    predicted_label = model.predict(new_question_vectorized)
    predicted_bloom_level = label_encoder.inverse_transform(predicted_label)
    return predicted_bloom_level[0]

def index(request):
    if request.method == 'POST':
        question = request.POST['question']
        bloom_level = predict_bloom_level(question)
        return render(request, 'bloom_app/index.html', {'bloom_level': bloom_level})
    return render(request, 'bloom_app/index.html')

@api_view(['POST'])
def predict_bloom_level_api(request):
    try:
        serializer = QuestionSerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            bloom_level = predict_bloom_level(question)
            
            response_data = {
                'bloom_level': bloom_level,
                'confidence': model.predict_proba(vectorizer.transform([question]).toarray()).max()
            }
            
            response_serializer = PredictionResponseSerializer(data=response_data)
            if response_serializer.is_valid():
                return Response(response_serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

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