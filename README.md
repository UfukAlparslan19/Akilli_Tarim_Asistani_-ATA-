🌱 ATA - Akıllı Tarım Asistanı

Yapay Zeka Destekli Bitki Hastalığı Teşhis ve Dijital Zirai Danışmanlık Sistemi

Tarımda verimi artırmak ve ürün kaybını teknolojiyle durdurmak için geliştirildi.




📖 Proje Hakkında

Akıllı Tarım Asistanı (ATA), çiftçilerin tarlalarındaki ürün hastalıklarını saniyeler içinde, yüksek doğrulukla tespit etmelerini sağlayan modern bir yapay zeka sistemidir. Sadece hastalığı bulmakla kalmaz; tespit edilen hastalığa, mahsulün türüne ve bitkinin boyuna göre Google Gemini AI gücüyle çiftçiye özel, anlaşılır ve hemen uygulanabilir bir Acil Zirai Eylem Planı sunar.

✨ Öne Çıkan Özellikler

📸 Derin Öğrenme ile Teşhis: Gelişmiş EfficientNet-B4 mimarisi ile 12 farklı tarımsal üründe mikroskobik doğrulukta teşhis.

🧠 Açıklanabilir Yapay Zeka (XAI): Grad-CAM algoritması sayesinde modelin kararı neye göre verdiği görselleştirilir. Yapraktaki hastalıklı bölgeler bir "röntgen" gibi işaretlenir.

🎯 Test Zamanı Veri Artırımı (TTA): Görüntü tahmin aşamasında simetrik varyasyonlar üretilerek kararsızlıklar giderilir, teşhisin kesinliği artırılır.

💬 Üretken YZ Ziraat Mühendisi: Tespit edilen hastalığı analiz eden Gemini 1.5 Flash, çiftçiye doğal ve destekleyici bir dille kültürel/kimyasal mücadele yöntemlerini listeler.

📊 Otomatik Eğitim ve Loglama: Yeni veriler eklendiğinde otonom olarak öğrenen, durması gereken yeri (EarlyStopping) bilen ve tüm sonuçları estetik grafiklerle raporlayan akıllı eğitim döngüsü.




🏆 Model Performansı ve Kapsam

Modelimiz, gelişmiş Albumentations veri artırma teknikleri ve Karışık Hassasiyetli Eğitim (Mixed Precision) kullanılarak zorlu şartlar (ışık parlamaları, bozulmalar) altında dahi yüksek başarı gösterecek şekilde eğitilmiştir.

Mahsul Doğruluk Oranı     Mahsul Doğruluk Oranı
🍅 Domates,%99.49,        🍓 Çilek,%92.33
🌽 Mısır,%97.12,          🍏 Elma,%92.96
🌿 Soya Fasulyesi,%96.65, 🌾 Buğday,%92.81
🍇 Üzüm,%96.45,           🌾 Arpa,%92.59
🍑 Kayısı,%96.07,         🥔 Patates,%90.55
🍚 Pirinç,%94.76,         🍑 Şeftali,%85.58




📁 Klasör Hiyerarşisi

Projenin kendi cihazınızda sorunsuz çalışması için önerilen dizin yapısı:

ATA-Projesi/
├── data/
│   └── raw/                   # Mahsul klasörlerinin (saglikli, hastalikli) bulunduğu dizin
├── grafikler/                 # Eğitim sürecinde otomatik üretilen başarı grafikleri
├── gradcam_ornekleri/         # XAI (Grad-CAM) çıktıları
├── egitim_otomatik.py         # 12 mahsulü sırayla eğiten akıllı eğitim scripti
├── test_gui.py                # Masaüstü Kullanıcı Arayüzü (Tkinter + Gemini AI)
├── README.md                  # Bu dosya
└── requirements.txt           # Bağımlılıklar




⚙️ Kurulum ve Kullanım Başlangıcı

1. Depoyu Klonlayın ve Gerekli Paketleri Kurun

Kodu bilgisayarınıza indirdikten sonra, proje dizininde bir terminal açarak gerekli yapay zeka ve görüntü işleme kütüphanelerini yükleyin:

git clone [https://github.com/KULLANICI_ADINIZ/ATA-Projesi.git](https://github.com/KULLANICI_ADINIZ/ATA-Projesi.git)
cd ATA-Projesi
pip install -r requirements.txt


(Eğer requirements.txt kullanmıyorsanız: pip install torch torchvision opencv-python numpy albumentations matplotlib grad-cam google-generativeai tqdm)

2. Yapay Zeka Modelini Eğitmek (Geliştiriciler İçin)

Kendi tarımsal verilerinizle (veya sağlanan veri setiyle) modeli sıfırdan eğitmek için:

python egitim_otomatik.py


Bu komut; klasördeki tüm bitkileri sırayla gezecek, eğitecek, en iyi .pth model ağırlıklarını kaydedecek ve sonuç raporlarını üretecektir.

3. Akıllı Asistanı Başlatmak (Son Kullanıcı İçin)

Çiftçilerin kullanacağı, fotoğraf yükleyip Gemini AI'dan tavsiye alınan görsel arayüzü başlatmak için:

python test_gui.py


⚠️ Çok Önemli: Danışmanlık özelliğinin çalışabilmesi için test_gui.py kodunun üst kısmında yer alan GEMINI_API_KEY değişkenine kendi Google AI Studio API anahtarınızı eklemeyi unutmayın!
