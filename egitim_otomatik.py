# egitim_otomatik.py (1.000'DE DURAN, ESTETİK GRAFİKLİ, GRAD-CAM'Lİ VE SONUÇ KAYDEDEN FİNAL SÜRÜM)
import warnings, os, sys, random, numpy as np, cv2, torch, torch.nn as nn, torch.optim as optim
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast

# --- GRAD-CAM VE GRAFİK İÇİN KÜTÜPHANELER ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# ---------------------------------------------

def worker_silencer(worker_id):
    sys.stderr = open(os.devnull, 'w')

warnings.filterwarnings("ignore")

# ------------------------
# 1) Veri Artırma
# ------------------------
IMAGE_BOYUTU = 300
train_transform = A.Compose([
    A.Resize(height=IMAGE_BOYUTU, width=IMAGE_BOYUTU),
    A.RandomRotate90(p=0.5), 
    A.HorizontalFlip(p=0.5), 
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2), 
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20), 
        A.RandomGamma(),
    ], p=0.7),
    A.OneOf([
        A.OpticalDistortion(p=0.3), 
        A.GridDistortion(p=0.1), 
        A.GaussNoise(var_limit=(10.0, 50.0)),
    ], p=0.3),
    A.CoarseDropout(max_holes=12, max_height=IMAGE_BOYUTU//6, max_width=IMAGE_BOYUTU//6, min_holes=2, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=IMAGE_BOYUTU, width=IMAGE_BOYUTU),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ------------------------
# 2) Veri Kümesi Sınıfı
# ------------------------
class PlantDataset(Dataset):
    def __init__(self, items, class_to_idx, augment=True, return_path=False):
        self.items = items; self.class_to_idx = class_to_idx; self.augment = augment; self.return_path = return_path
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label_name = self.items[idx]; label = self.class_to_idx[label_name]
        try:
            img_np = np.fromfile(path, dtype=np.uint8); img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if img is None: raise IOError(f"OpenCV dosyayı decode edemedi: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"HATA: {path}, Detay: {e}"); img = np.zeros((IMAGE_BOYUTU, IMAGE_BOYUTU, 3), dtype=np.uint8)
        
        transform = train_transform if self.augment else val_transform
        img_tensor = transform(image=img)['image']
        
        if self.return_path: return img_tensor, torch.tensor(label, dtype=torch.long), path
        return img_tensor, torch.tensor(label, dtype=torch.long)

# ------------------------
# 3) Model
# ------------------------
def build_model(n_classes, dropout_rate=0.5):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(model.classifier[1].in_features, n_classes)
    )
    return model

# ------------------------
# 4) Veri Hazırlama
# ------------------------
def prepare_dataset_for_crop(config, val_split=0.2):
    train_items, val_items = [], []
    class_names = set()
    crop_data_root = config['data_root']; mahsul_adi = config['mahsul_adi']
    print(f"'{mahsul_adi}' için veri hazırlanıyor... Sadece '{mahsul_adi}_' ile başlayan klasörler işlenecek.")
    
    if not os.path.exists(crop_data_root):
        print(f"⚠️ DİKKAT: '{crop_data_root}' klasörü bulunamadı. Bu mahsul atlanıyor...")
        return [], [], [], {}

    for status_folder in sorted(os.listdir(crop_data_root)):
        status_path = os.path.join(crop_data_root, status_folder)
        if not os.path.isdir(status_path): continue
        if status_folder == 'hastalikli':
            for disease_folder in sorted(os.listdir(status_path)):
                if disease_folder.startswith(mahsul_adi + "_"):
                    disease_path = os.path.join(status_path, disease_folder)
                    if not os.path.isdir(disease_path): continue
                    label_name = disease_folder; class_names.add(label_name)
                    files = glob(os.path.join(disease_path, '*.jpg')) + glob(os.path.join(disease_path, '*.png')) + glob(os.path.join(disease_path, '*.jpeg'))
                    random.shuffle(files); split_idx = int(len(files) * (1 - val_split))
                    train_items += [(f, label_name) for f in files[:split_idx]]
                    val_items += [(f, label_name) for f in files[split_idx:]]
        elif status_folder == 'saglikli':
            label_name = 'saglikli'; class_names.add(label_name)
            files = glob(os.path.join(status_path, '*.jpg')) + glob(os.path.join(status_path, '*.png')) + glob(os.path.join(status_path, '*.jpeg'))
            random.shuffle(files); split_idx = int(len(files) * (1 - val_split))
            train_items += [(f, label_name) for f in files[:split_idx]]
            val_items += [(f, label_name) for f in files[split_idx:]]
            
    class_names = sorted(list(class_names)); class_to_idx = {name: i for i, name in enumerate(class_names)}
    return train_items, val_items, class_names, class_to_idx

# ------------------------
# 5) Validation
# ------------------------
def validate_with_tta(model, loader, criterion, device):
    model.eval(); correct, total = 0, 0; total_val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Doğrulama (TTA)"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs_orig = model(imgs)
            loss = criterion(outputs_orig, labels)
            total_val_loss += loss.item() * imgs.size(0)
            outputs_h = model(torch.flip(imgs, [3])); outputs_v = model(torch.flip(imgs, [2]))
            avg_outputs = (torch.softmax(outputs_orig, 1) + torch.softmax(outputs_h, 1) + torch.softmax(outputs_v, 1)) / 3.0
            preds = torch.argmax(avg_outputs, dim=1)
            correct += (preds == labels).sum().item(); total += labels.size(0)
    return correct / total, total_val_loss / len(loader.dataset)

# --- 🚀 GRAD-CAM ÇİZME VE KAYDETME FONKSİYONU 🚀 ---
def save_gradcam_examples(model, class_names, device, mahsul_adi):
    print(f"👁️ {mahsul_adi.upper()} için Grad-CAM örnekleri oluşturuluyor...")
    os.makedirs('gradcam_ornekleri', exist_ok=True)
    
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    gradcam_dataset = PlantDataset(val_items, class_to_idx, augment=False, return_path=True)
    batch_size = min(4, len(gradcam_dataset))
    if batch_size == 0:
        return

    gradcam_loader = DataLoader(gradcam_dataset, batch_size=batch_size, shuffle=True)
    imgs_tensor, labels_tensor, paths = next(iter(gradcam_loader))
    imgs_tensor = imgs_tensor.to(device)
    
    plt.figure(figsize=(16, 8))
    
    for i in range(batch_size):
        img_np = np.fromfile(paths[i], dtype=np.uint8); img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (IMAGE_BOYUTU, IMAGE_BOYUTU))
        img_normalized = img_rgb.astype(float) / 255.0
        
        model.eval()
        with torch.no_grad():
            output = model(imgs_tensor[i].unsqueeze(0))
            pred_idx = torch.argmax(output, dim=1).item()
            pred_label = class_names[pred_idx]
            true_label = class_names[labels_tensor[i].item()]
        
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=imgs_tensor[i].unsqueeze(0), targets=targets)[0, :]
        cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        
        plt.subplot(2, 4, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"Orijinal\nGerçek: {true_label}")
        plt.axis('off')
        
        plt.subplot(2, 4, i + 5)
        plt.imshow(cam_image)
        plt.title(f"Grad-CAM Röntgeni\nTahmin: {pred_label}")
        plt.axis('off')
        
    plt.tight_layout()
    gradcam_yolu = os.path.join('gradcam_ornekleri', f"{mahsul_adi}_gradcam_ornek.png")
    plt.savefig(gradcam_yolu, dpi=300)
    plt.close()
# ----------------------------------------------------

# ------------------------
# 6) Eğitim Döngüsü
# ------------------------
def train_pipeline(config):
    global val_items, class_to_idx
    
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Cihaz: {device.upper()}")
    train_items, val_items, class_names, class_to_idx = prepare_dataset_for_crop(config)
    n_classes = len(class_names)
    
    if not train_items or not val_items: 
        print("⚠️ Veri bulunamadı veya yetersiz. Bu mahsul atlanıyor.")
        return 0.0
        
    print(f"--- {config['mahsul_adi'].upper()} Modeli Eğitiliyor ---"); print(f"Toplam Sınıf Sayısı: {n_classes}")

    train_dataset = PlantDataset(train_items, class_to_idx, augment=True)
    val_dataset = PlantDataset(val_items, class_to_idx, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True, worker_init_fn=worker_silencer)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True, worker_init_fn=worker_silencer)

    model = build_model(n_classes, dropout_rate=config.get('dropout', 0.5)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
    scaler = GradScaler()
    best_acc, patience = 0.0, 0

    history_train_loss = []
    history_val_loss = []
    history_val_acc = []

    print("Eğitim Başlatılıyor...")
    for epoch in range(config['epochs']):
        model.train(); total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item() * imgs.size(0)

        epoch_train_loss = total_loss / len(train_dataset)
        val_acc, val_loss = validate_with_tta(model, val_loader, criterion, device)
        scheduler.step()
        
        # --- 🚀 HEM DURDURAN HEM DE GRAFİĞİ TEMİZ TUTAN MANTIK 🚀 ---
        if val_acc >= 0.995: 
            print(f"⚠️ DİKKAT: Model %100 doğruluğa (ezbere) ulaştı!")
            print(f"🛑 Eğitim anında DURDURULDU. Bu sahte tepe noktası grafiğe çizilmeyecek.")
            break 
                
        history_train_loss.append(epoch_train_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc) 
        # ---------------------------------------------------------------

        print(f"Epoch {epoch+1}: TrainLoss={epoch_train_loss:.4f} | ValLoss={val_loss:.4f} | ValAcc (TTA)={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), config['model_path'])
            print(f"✅ [{config['mahsul_adi'].upper()}] Yeni en iyi model kaydedildi (Doğruluk: {best_acc:.4f})")
        else:
            patience += 1
            print(f"Sabır: {patience}/{config['patience']}")

        if patience >= config['patience']:
            print("🔴 Erken durdurma tetiklendi."); break
            
    with open(config['classes_path'], "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    # --- GRAFİK ÇİZME VE KAYDETME BÖLÜMÜ ---
    print(f"📊 {config['mahsul_adi'].upper()} için eğitim grafiği oluşturuluyor...")
    os.makedirs('grafikler', exist_ok=True)
    
    if len(history_val_acc) > 0:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history_train_loss, label='Eğitim Kaybı', color='#3b82f6', marker='o')
        plt.plot(history_val_loss, label='Doğrulama Kaybı', color='#ef4444', marker='o')
        plt.title(f"{config['mahsul_adi'].upper()} - Model Kaybı (Loss)")
        plt.xlabel('Epoch'); plt.ylabel('Kayıp'); plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history_val_acc, label='Doğrulama Oranı', color='#10b981', marker='o')
        plt.title(f"{config['mahsul_adi'].upper()} - Doğruluk Oranı")
        plt.xlabel('Epoch'); plt.ylabel('Doğruluk')
        plt.ylim(0.0, 1.01) 
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        
        plt.tight_layout()
        grafik_yolu = os.path.join('grafikler', f"{config['mahsul_adi']}_egitim_grafigi.png")
        plt.savefig(grafik_yolu, dpi=300)
        plt.close()
        print(f"✅ Estetik Grafik güncellendi: {grafik_yolu}")

    # --- GRAD-CAM ÖRNEKLERİNİ ÇİZ ---
    if os.path.exists(config['model_path']):
        model.load_state_dict(torch.load(config['model_path']))
    save_gradcam_examples(model, class_names, device, config['mahsul_adi'])

    return best_acc

# ------------------------
# 7) Ana Program
# ------------------------
if __name__ == "__main__":
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    
    # KİRAZ MAHSULÜ LİSTEDEN ÇIKARILDI, KAYISI EKLİ
    MAHSULLER = ['kayisi', 'uzum', 'seftali', 'soya fasulyesi', 'cilek', 'elma', 'bugday', 'arpa', 'misir', 'pirinc', 'domates', 'patates']
    sonuclar = {}

    for mahsul_adi in MAHSULLER:
        print(f"\n" + "="*50); print(f"SIRADAKİ EĞİTİM: {mahsul_adi.upper()}"); print("="*50 + "\n")
        CONFIG = {
            'mahsul_adi': mahsul_adi,
            'data_root': os.path.join('data', 'raw', mahsul_adi),
            'epochs': 10,
            'patience': 5,
            'batch_size': 4,
            'lr': 1e-4,
            'weight_decay': 3e-3,
            'num_workers': 4,
            'label_smoothing': 0.1,
            'dropout': 0.5,
            'model_path': f'best_model_b4_{mahsul_adi}.pth',
            'classes_path': f'class_names_b4_{mahsul_adi}.txt'
        }
        
        if mahsul_adi in ['seftali', 'şeftali', 'pirinc', 'biber', 'kayisi']:
            print(f"⚙️ {mahsul_adi.upper()} için Yüksek Regularizasyon Ayarları Uygulanıyor.")
            CONFIG['weight_decay'] = 8e-3
            CONFIG['patience'] = 7
            CONFIG['label_smoothing'] = 0.15
            CONFIG['dropout'] = 0.6
            
        if mahsul_adi in ['misir']:
            print("🌽 Stabilite modu: Worker sayısı düşürülüyor.")
            CONFIG['num_workers'] = 2
            
        acc = train_pipeline(CONFIG)
        sonuclar[mahsul_adi] = acc

    print("\n\n" + "="*50); print("🎉 TÜM EĞİTİMLER TAMAMLANDI! 🎉"); print("="*50 + "\n")
    print("--- 📊 EĞİTİM ÖZETİ (En İyi Doğruluk Oranları) ---")
    
    if not sonuclar:
        print("Hiçbir model eğitilmedi.")
    else:
        # --- 🚀 YENİ EKLENEN OTOMATİK KAYIT SİSTEMİ 🚀 ---
        ozet_metni = "--- ATA PROJESI EĞİTİM ÖZETİ ---\n\n"
        for mahsul, acc in sonuclar.items():
            satir = f"- {mahsul.upper()}: %{acc * 100:.4f}\n"
            print(satir.strip())
            ozet_metni += satir
            
        # Sonuçları txt dosyasına yaz
        with open("egitim_sonuclari_ozeti.txt", "w", encoding="utf-8") as f:
            f.write(ozet_metni)
            
        print("\n✅ Harika! Tüm bu sonuçlar aynı klasördeki 'egitim_sonuclari_ozeti.txt' dosyasına kalıcı olarak kaydedildi!")