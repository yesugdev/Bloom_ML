# 🎯 Сэтгэл хөдлөл ба Блүмийн ангилалын таамаглал API

## 📝 Төслийн тухай
Энэхүү төсөл нь дараах 2 үндсэн үйлдлийг гүйцэтгэнэ:
- Боловсролын асуултуудын Блүмийн түвшинг таамаглах (1-6)
- Текстэн дэх сэтгэл хөдлөлийг шинжлэх (уйтгар(0), баяр(1), хайр(2), уур(3), айдас(4))
  
- Блүмийн ангилал: 8000 асуулт бүхий датасет дээр сургасан
- Сэтгэл хөдлөл: 2000 өгүүлбэр бүхий датасет дээр сургасан
- Алгоритм: RandomForest Classifier
- Нарийвчлал: ~68%

## ⚙️ Суулгах заавар
```bash
# Төслийг татах
git clone https://github.com/yesugdev/Bloom_ML
cd Bloom_ML

# Шаардлагатай сангуудыг суулгах
pip install -r requirements.txt

cd bloom_project
# Өгөгдлийн сан үүсгэх
python manage.py migrate

# Серверийг ажиллуулах
python manage.py runserver
```

🚀 Боломжууд
✨ REST API үйлчилгээ
📊 Excel макро интеграци
📦 Багц боловсруулалт
💾 Excel-рүү экспортлох
🔌 API Үйлчилгээ
Блүмийн ангилал
```bash
curl -X POST ^
  http://localhost:8000/api/predict/ ^
  -H "Content-Type: application/json" ^
  -H "Accept: application/json" ^
  -d "{\"question\":\"Жишээ асуулт?\"}"
```

```bash
curl -X POST http://localhost:8000/api/emotion/ -H "Content-Type: application/json" -d "{\"text\":\"Би зүгээр л уйтгартай, хөхөрч байна\"}"
```
📈 Excel интеграци
Excel програм нээх
Alt + F11 товч дарж VBA editor нээх
macro_for_excel.txt файлыг импортлох
A баганад текст оруулах
Макрог ажиллуулах
```
🛠️ Ашигласан технологиуд
Python 3.13+
Django 4.0+
scikit-learn
pandas
Microsoft Excel
📋 Шаардлагатай сангууд
```
👨‍💻 Зохиогч
Yesug Bekhbold

