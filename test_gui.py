# test_gui_final.py (SON EĞİTİM KODUNA TAM UYUMLU FİNAL VERSİYON)

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import google.generativeai as genai

# --------------------------------------------------------------------------------------
# ⚠️ DİKKAT: API key süreli (veya kotalı) olduğu için test.py dosyasını sorunsuz 
# çalıştırmak adına lütfen Google AI Studio'dan kendi API key'inizi oluşturup aşağıya yapıştırınız.
# --------------------------------------------------------------------------------------
GEMINI_API_KEY = "BURAYA_YAPISTIRIN" 

# -----------------------------------------------------
# 1) AYARLAR (egitim_otomatik.py ile tam uyumlu)
# -----------------------------------------------------
IMAGE_SIZE = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ***GÜNCELLEME: Eğitim kodundaki tam listenin aynısı eklendi***
MAHSULLER = ['kayisi', 'uzum', 'seftali', 'soya fasulyesi', 'cilek', 'elma', 'bugday', 'arpa', 'misir', 'pirinc', 'domates', 'patates'] 

# -----------------------------------------------------
# 2) GÖRÜNTÜ İŞLEME VE MODEL MİMARİSİ
# -----------------------------------------------------
# Preprocess transform, eğitim kodundaki val_transform ile birebir aynıdır.
preprocess_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def build_model(n_classes):
    # Model yapısı, eğitim kodundaki `build_model` fonksiyonu ile tam uyumludur.
    model = models.efficientnet_b4(weights=None) 
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(model.classifier[1].in_features, n_classes)
    )
    return model

# -----------------------------------------------------
# 3) TAHMİN FONKSİYONU (TTA İLE)
# -----------------------------------------------------
def predict_disease(image_path, model, class_names):
    try:
        # Resim okuma yöntemi eğitim kodu ile uyumlu
        img_np = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None: raise IOError("OpenCV dosyayı okuyamadı.")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        transformed = preprocess_transform(image=img_rgb)
        img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            # TTA (Test Time Augmentation) uygulaması
            outputs_h = model(torch.flip(img_tensor, [3]))
            outputs_v = model(torch.flip(img_tensor, [2]))
            
            # 3 çıktının ortalaması alınarak kesinlik artırılır
            avg_probabilities = (torch.softmax(outputs, dim=1) + torch.softmax(outputs_h, dim=1) + torch.softmax(outputs_v, dim=1)) / 3.0
            
            confidence, top_class_idx = torch.topk(avg_probabilities[0], 1)
            pred_class_name = class_names[top_class_idx.item()]
            confidence_percent = confidence.item() * 100
            
            return pred_class_name, confidence_percent
    except Exception as e:
        print(f"Hastalık teşhisi sırasında hata: {e}")
        return "Hata!", 0.0

# -----------------------------------------------------
# 4) LLM FONKSİYONU
# -----------------------------------------------------
def get_llm_advice(mahsul_adi, hastalik_adi, mahsul_boyu):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "BURAYA_YAPISTIRIN":
        return "⚠️ HATA: Lütfen kodun en üstündeki GEMINI_API_KEY değişkenine geçerli bir Google Gemini API anahtarı girin."
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        prompt = f"""
        Sen, "AgriChat AI" adlı bir mobil uygulamanın yardımsever ve uzman bir ziraat mühendisisin.
        Bir çiftçi, tarlasındaki bir bitkinin fotoğrafını çekti ve bazı bilgiler verdi. 
        Analiz sonuçları aşağıdadır:
        
        - Mahsul Türü: {mahsul_adi}
        - Tespit Edilen Hastalık/Durum: {hastalik_adi}
        - Çiftçinin Bildirdiği Bitki Boyu: {mahsul_boyu} cm
        
        Lütfen bu bilgilere dayanarak, çiftçiye doğrudan hitap ederek, sanki onunla konuşuyormuş gibi bir tavsiye raporu hazırla. Raporun aşağıdaki bölümleri içermeli:
        
        1.  **Teşhis:** Anlaşılır bir dille hastalığın ne olduğunu söyle. ("Merhaba, fotoğrafını inceledim ve bitkinizde...")
        2.  **Boy Değerlendirmesi:** Verilen bitki boyunun, o mahsul için normal olup olmadığını, bu hastalığın bitkinin gelişimini (boyunu) etkileyip etkilemediğini kısaca yorumla.
        3.  **Acil Eylem Planı:** Hastalığın yayılmasını önlemek için çiftçinin **hemen şimdi** yapması gereken en önemli 2-3 adımı maddeler halinde sırala.
        4.  **Mücadele Yöntemleri:** Hem kimyasal (ilaç) hem de kültürel (doğal) mücadele yöntemlerini ayrı başlıklar altında, basit ve uygulanabilir şekilde anlat.
        5.  **Önemli Bir İpucu:** Son olarak, bu hastalıkla ilgili çiftçinin aklında kalması gereken altın değerinde bir ipucu ver.
        
        Lütfen teknik terimlerden kaçın, samimi ve destekleyici bir dil kullan.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_message = f"Uzman tavsiyesi alınırken bir hata oluştu. Lütfen aşağıdakileri kontrol edin:\n\n1. İnternet bağlantınız var mı?\n2. API anahtarınız doğru ve geçerli mi (Süresi dolmuş olabilir)?\n3. Google Cloud projenizde 'Generative Language API' etkin mi?\n\nTeknik Hata Detayı:\n{e}"
        return error_message

# -----------------------------------------------------
# 5) TKINTER ARAYÜZÜ
# -----------------------------------------------------
class DigitalFarmApp:
    def __init__(self, root):
        self.root = root; 
        self.root.title("AgriChat AI - Akıllı Zirai Danışman"); 
        self.root.geometry("800x800")
        self.model = None; self.class_names = None; self.selected_mahsul = tk.StringVar(); self.image_path = None
        
        step1_frame = ttk.LabelFrame(root, text=" Adım 1: Mahsulü Seçin ", padding=(10, 10)); step1_frame.pack(fill=tk.X, padx=10, pady=5)
        self.mahsul_menu = ttk.Combobox(step1_frame, textvariable=self.selected_mahsul, values=MAHSULLER, state="readonly", font=("Helvetica", 12)); self.mahsul_menu.pack(fill=tk.X, padx=5, pady=5)
        self.mahsul_menu.bind("<<ComboboxSelected>>", self.load_model_for_mahsul)
        step2_frame = ttk.LabelFrame(root, text=" Adım 2: Bilgileri Girin ", padding=(10, 10)); step2_frame.pack(fill=tk.X, padx=10, pady=5)
        self.btn_select_image = ttk.Button(step2_frame, text="Hastalıklı Bitki Fotoğrafı Yükle", command=self.select_image, state=tk.DISABLED); self.btn_select_image.pack(fill=tk.X, padx=5, pady=5)
        self.lbl_image_path = ttk.Label(step2_frame, text="Henüz fotoğraf seçilmedi.", anchor=tk.W); self.lbl_image_path.pack(fill=tk.X, padx=5)
        step3_frame = ttk.LabelFrame(root, text=" Adım 3: Analiz Et ", padding=(10, 10)); step3_frame.pack(fill=tk.X, padx=10, pady=5)
        self.btn_analyze = ttk.Button(step3_frame, text="Akıllı Danışmandan Tavsiye Al", command=self.analyze, state=tk.DISABLED); self.btn_analyze.pack(fill=tk.X, padx=5, pady=5)
        result_frame = ttk.LabelFrame(root, text=" Uzman Raporu ", padding=(10, 10)); result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.txt_result = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, font=("Helvetica", 12), state=tk.DISABLED); self.txt_result.pack(fill=tk.BOTH, expand=True)
        self.status_bar = ttk.Label(root, text="Başlamak için bir mahsul seçin.", relief=tk.SUNKEN, anchor=tk.W); self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model_for_mahsul(self, event=None):
        mahsul = self.selected_mahsul.get(); self.status_bar.config(text=f"'{mahsul.upper()}' modeli yükleniyor..."); self.root.update_idletasks()
        # Model ve sınıf dosyası adları eğitim kodu ile uyumlu.
        model_path = f'best_model_b4_{mahsul}.pth'; classes_path = f'class_names_b4_{mahsul}.txt'
        if not all(os.path.exists(p) for p in [model_path, classes_path]): messagebox.showerror("Hata", f"'{mahsul}' için gerekli model dosyaları bulunamadı! (Beklenen: {model_path} ve {classes_path})"); self.reset_ui(); return
        try:
            with open(classes_path, "r", encoding="utf-8") as f: self.class_names = [line.strip() for line in f.readlines()]
            self.model = build_model(len(self.class_names)); self.model.load_state_dict(torch.load(model_path, map_location=DEVICE)); self.model.to(DEVICE); self.model.eval()
            self.status_bar.config(text=f"✅ '{mahsul.upper()}' modeli hazır. Lütfen bir fotoğraf yükleyin."); self.btn_select_image.config(state=tk.NORMAL); self.image_path = None
            self.lbl_image_path.config(text="Henüz fotoğraf seçilmedi."); self.btn_analyze.config(state=tk.DISABLED)
        except Exception as e: messagebox.showerror("Yükleme Hatası", f"Model yüklenirken hata oluştu: {e}"); self.reset_ui()

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Bir resim seçin", filetypes=[("Image Files", "*.jpg *.jpeg *.png")]);
        if not file_path: return
        self.image_path = file_path; self.lbl_image_path.config(text=os.path.basename(file_path)); self.btn_analyze.config(state=tk.NORMAL); self.status_bar.config(text="Fotoğraf seçildi. Analiz için butona basın.")
    
    def analyze(self):
        if not self.image_path: messagebox.showwarning("Eksik Bilgi", "Lütfen önce bir fotoğraf yükleyin."); return
        mahsul_boyu = simpledialog.askinteger("Mahsul Boyu", "Lütfen mahsulünüzün ortalama boyunu santimetre (cm) olarak girin:", parent=self.root)
        if mahsul_boyu is None: return
        self.status_bar.config(text="Hastalık teşhis ediliyor..."); self.root.update_idletasks()
        
        # Sadece temiz 2 değişken dönüyor
        hastalik_adi, confidence = predict_disease(self.image_path, self.model, self.class_names)

        if "Hata!" in hastalik_adi: messagebox.showerror("Teşhis Hatası", "Fotoğraf analiz edilirken bir sorun oluştu."); return
        
        self.status_bar.config(text=f"Teşhis: {hastalik_adi} (%{confidence:.2f}). Akıllı danışmandan tavsiye alınıyor..."); self.root.update_idletasks()
        advice = get_llm_advice(self.selected_mahsul.get(), hastalik_adi, mahsul_boyu)
        self.txt_result.config(state=tk.NORMAL); self.txt_result.delete(1.0, tk.END); self.txt_result.insert(tk.END, advice); self.txt_result.config(state=tk.DISABLED)
        self.status_bar.config(text="✅ Uzman raporu başarıyla oluşturuldu.")
        
    def reset_ui(self):
        self.status_bar.config(text="Başlamak için bir mahsul seçin."); self.btn_select_image.config(state=tk.DISABLED); self.btn_analyze.config(state=tk.DISABLED)
        self.image_path = None; self.lbl_image_path.config(text="Henüz fotoğraf seçilmedi.")

# -----------------------------------------------------
# 6) UYGULAMAYI BAŞLATMA
# -----------------------------------------------------
if __name__ == "__main__":
    main_window = tk.Tk()
    app = DigitalFarmApp(main_window)
    main_window.mainloop()