import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # Сургалтын явцыг хянахын тулд tqdm ашиглах

# 1-р алхам: Excel файлыг ачааллах
file_path = '8000_Блоом_түвшин_датасэт.xlsx'  # Өөрийн файлын замыг оруулна уу
df = pd.read_excel(file_path)

# 2-р алхам: Өгөгдлийн бүрдлийг шалгах
print(df.head())  # Өгөгдлийн 'Question' болон 'Label' баганууд байгаа эсэхийг шалгах

# 3-р алхам: 'Label' баганыг Bloom-ийн ангиллын дагуу тоон утгад хөрвүүлэх
label_encoder = LabelEncoder()
df['Label_Encoded'] = label_encoder.fit_transform(df['Category'])  # Ангиллын утгуудыг кодлох

# 4-р алхам: Асуултуудыг тоон өгөгдөл болгон хувиргахын тулд TF-IDF векторчлол ашиглах
vectorizer = TfidfVectorizer(max_features=500)  # 500 онцлог шинж тэмдэгт хязгаарлах
X = vectorizer.fit_transform(df['Questions']).toarray()  # Асуултуудыг тоон вектор болгон хөрвүүлэх

# 5-р алхам: Шинж чанар (X) болон зорилтот хувьсагч (y)-г тодорхойлох
y = df['Label_Encoded']  # Зорилтот хувьсагч (Bloom-ийн ангилал)

# 6-р алхам: Өгөгдлийг сургалтын (train) болон шалгалтын (test) хэсэгт хуваах
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7-р алхам: Санамсаргүй ой (Random Forest) загвар сургах
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Сургалтын явцыг харуулахын тулд tqdm ашиглах
print("Загвар сургалт эхэлж байна...")
for _ in tqdm(range(1), desc="Сургалтын явц", total=1):
    model.fit(X_train, y_train)  # Загварыг сургах

# 8-р алхам: Сорилтын өгөгдөл дээр таамаглал гаргах
y_pred = model.predict(X_test)

# 9-р алхам: Загварын гүйцэтгэлийг үнэлэх (нарийвчлал)
accuracy = accuracy_score(y_test, y_pred)
print(f"Нарийвчлал: {accuracy}")

# 10-р алхам: Шинэ асуултад Bloom-ийн түвшин таамаглах функц
def predict_bloom_level(question):
    new_question_vectorized = vectorizer.transform([question]).toarray()  # Шинэ асуултыг тоон өгөгдөл болгон хөрвүүлэх
    predicted_label = model.predict(new_question_vectorized)  # Bloom-ийн түвшинг таамаглах
    predicted_bloom_level = label_encoder.inverse_transform(predicted_label)  # Тоон утгийг анхны ангилал руу хөрвүүлэх
    return predicted_bloom_level[0]  # Bloom-ийн түвшинг буцаах

# 11-р алхам: Хэрэглэгчээс тасралтгүй асуулт авах
while True:
    question = input("Асуултаа оруулна уу (гарахын тулд 'exit' гэж бичнэ үү): ")
    if question.lower() == 'exit':
        break
    bloom_level = predict_bloom_level(question)
    print(f"Асуултын Bloom-ийн таамагласан түвшин: {bloom_level}")
