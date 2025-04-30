import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Emotion mapping
emotion_map = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

# 1. Data Preparation Function
def prepare_data(filepath='emotion_data.csv', samples_per_class=2000):
    """Load and balance the dataset"""
    df = pd.read_csv(filepath)
    
    # Sample balanced data
    sampled_data = []
    for label in emotion_map.keys():
        class_samples = df[df['label'] == label].sample(n=samples_per_class, random_state=42)
        sampled_data.append(class_samples)
    
    return pd.concat(sampled_data)

# 2. Text Preprocessing
def preprocess_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# 3. Training Function
def train_model():
    """Train and save the emotion classifier"""
    print("Loading and preparing data...")
    df = prepare_data()
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Feature engineering
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7
    )
    X = tfidf.fit_transform(df['processed_text'])
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    print("Training Random Forest model...")
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=30,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred, target_names=list(emotion_map.values())))
    
    # Save model
    joblib.dump(rf, 'emotion_classifier_rf.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print("\nModel saved to 'emotion_classifier_rf.pkl'")
    
    # Visualizations
    generate_visualizations(rf, tfidf, X_test, y_test, df)

# 4. Visualization Function
def generate_visualizations(model, vectorizer, X_test, y_test, df):
    """Generate evaluation visualizations"""
    print("\nGenerating visualizations...")
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(emotion_map.values()),
                yticklabels=list(emotion_map.values()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 6))
    plt.title('Top 20 Important Features')
    plt.barh(range(20), importances[indices], color='skyblue')
    plt.yticks(range(20), vectorizer.get_feature_names_out()[indices])
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Word Clouds
    plt.figure(figsize=(15, 10))
    for i, (num, name) in enumerate(emotion_map.items()):
        plt.subplot(2, 3, i+1)
        text = ' '.join(df[df['label'] == num]['processed_text'])
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(text)
        plt.imshow(wordcloud)
        plt.title(name)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('wordclouds.png')
    plt.close()

# 5. Prediction Interface
class EmotionPredictor:
    def __init__(self):
        try:
            self.model = joblib.load('emotion_classifier_rf.pkl')
            self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
        except:
            raise Exception("Model files not found. Please train the model first.")
    
    def predict_emotion(self, text):
        """Predict emotion from text input"""
        processed_text = preprocess_text(text)
        text_vector = self.vectorizer.transform([processed_text])
        
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        
        return prediction, probabilities

# Main Application
def main():
    print("\n" + "="*50)
    print("EMOTION CLASSIFICATION SYSTEM".center(50))
    print("="*50)
    
    while True:
        print("\nOptions:")
        print("1. Train new model")
        print("2. Predict emotion from text")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            train_model()
        elif choice == '2':
            try:
                predictor = EmotionPredictor()
            except Exception as e:
                print(f"\nError: {e}")
                continue
                
            text = input("\nEnter text to analyze: ").strip()
            if not text:
                print("Please enter some text")
                continue
                
            pred, probs = predictor.predict_emotion(text)
            
            print("\n" + "-"*50)
            print(f"Input: '{text}'")
            print(f"Predicted Emotion: {pred} ({emotion_map[pred]})")
            print("\nConfidence Scores:")
            for i, (num, name) in enumerate(emotion_map.items()):
                print(f"{name}: {probs[i]*100:.1f}%")
            print("-"*50)
        elif choice == '3':
            print("\nExiting program...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
