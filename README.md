# YZV416E Computer Vision - Project: SAM-Med

**Adapting Segment Anything – Decoder Fine-tuning on Domain-Specific Datasets (Medical Image Segmentation)**

**Ekip:** SAM-Med
- **Ömer Faruk Satık**: Core Codebase & Data Pipeline Lead
- **Bedirhan Öztürk**: Evaluation & Inference Lead
- **Abdullah Aydoğan**: Model Training & Optimization Lead

---

## 📌 Proje Özeti
Bu proje, Meta'nın **Segment Anything Model (SAM)** mimarisini (ViT-B), endoskopik görüntülerdeki poliplerin (gastrointestinal polipler) piksel seviyesinde bölütlenmesi amacıyla **Kvasir-SEG** veri seti üzerinde ince ayar (fine-tuning) yaparak tıp alanına adapte etmeyi amaçlamaktadır. Donanım sınırlarına (Örn: 16GB VRAM) saygı duymak adına, SAM'in ağır ViT kodlayıcısı (encoder) dondurulacak ve sadece hafif maske çözücüsü (decoder) eğitilecektir.

---

## 🚀 Kurulum ve Başlangıç Yönergesi (Anahtar Teslim)

Görevi devralan ekip arkadaşının (veya projeyi test edecek hocanın) projeyi kendi bilgisayarında veya Colab'da sıfırdan sorunsuz çalıştırabilmesi için aşağıdaki adımları sırasıyla uygulaması gerekmektedir.

### 0. Projenin Klonlanması
Öncelikle projemizin ana reposunu bilgisayarınıza (veya Colab'a) klonlayın ve klasörün içine girin:
```bash
git clone https://github.com/aydogn/Computer-Vision-Project.git
cd Computer-Vision-Project
```

### 1. Conda Ortamının Oluşturulması ve Aktif Edilmesi
Öncelikle çakışmaları önlemek için temiz bir Python sanal ortamı oluşturun:
```bash
conda create -n sam-med python=3.10 -y
conda activate sam-med
```

### 2. Kütüphanelerin Kurulumu
Gerekli olan Pytorch (CUDA destekli) ve tüm dış bağımlılıkları yüklemek için repository içerisinde yer alan `requirements.txt` dosyasını kullanın:
```bash
# Eğer ekran kartınız (GPU) varsa CUDA 12.6 sürümü için Pytorch kurulumu (Önerilen)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Diğer gereksinimleri (Albumentations, Segment-Anything vb.) kurun
pip install -r requirements.txt
```
*(Not: `requirements.txt` içerisinde yer alan `git+https://...` bağlantısı sayesinde Meta'nın **Segment-Anything** reposu arka planda otomatik olarak indirilip ortama kurulacaktır. Ayrı bir klonlama işlemi yapmanıza gerek yoktur!)*

### 3. Veri Seti ve Model Ağırlıklarının İndirilmesi
Veri setini manuel olarak indirmemek için hazırladığımız Python script'ini çalıştırın. Bu script Huggingface üzerinden Kvasir-SEG veri setini `kvasir-seg` klasörüne otomatik çıkartacaktır:
```bash
python download_data.py
```

Aynı şekilde SAM modelinin önceden eğitilmiş (Pre-trained) ViT-B ağırlığını indirmek için terminalden şu komutu çalıştırın (Otomatik olarak `checkpoints` klasörüne inecektir):
```powershell
mkdir checkpoints
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile checkpoints\sam_vit_b_01ec64.pth
```
*(Linux/Colab kullanıyorsanız `wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/` kullanabilirsiniz)*

### 4. Kurulumun Test Edilmesi
Data Pipeline'ın doğru çalıştığını (resim ve maskelerin boyutlandırılması, binarizasyon ve augmentation) doğrulamak için test script'ini çalıştırın:
```bash
python test_dataset.py
```
Bu komut, sorunsuz çalışırsa proje ana dizininde `test_output_v2.png` isimli örnek bir çıktı resmi üretecektir.

---

## 🗺️ Proje Yol Haritası (Roadmap) ve Mevcut Durum

Projenin **5 Aşamalı** bir uygulama planı bulunmaktadır. Şu ana kadar ilk 2 aşamanın veri mühendisliği kısımları tamamlanmıştır.

### ✅ Tamamlanan Aşamalar
* **Aşama 1: Ortam Kurulumu ve Veri Boru Hattı** (Ömer Faruk Satık)
  - GitHub reposu ve Conda ortamı ayarlandı.
  - Kvasir-SEG indirme scripti (`download_data.py`) yazıldı.
  - SAM ViT-B checkpoint'i sisteme entegre edildi.
  - Orijinal görüntü ile maskeleri okuyup eşleştiren PyTorch `Dataset` sınıfı yazıldı.
  - Maskeler binary (0 ve 1) tensor formatına zorlandı.
* **Aşama 2: Veri Ön İşleme (Data Preprocessing)** (Ömer Faruk Satık)
  - `torchvision` yerine `albumentations` kütüphanesine geçilerek senkronize dönüşümler kodlandı.
  - Polip aspect-ratio'sunu (en-boy oranını) korumak için `LongestMaxSize(1024)` ve siyah piksellerle `PadIfNeeded` eklendi.
  - Eğitim seti için `HorizontalFlip`, `VerticalFlip`, `RandomRotate90` ve `ColorJitter` ile online veri artırımı (augmentation) sağlandı.
  - SAM standartlarında (`mean`, `std`) ImageNet normalizasyonu yapıldı.
  - OOM hatalarını engellemek için `batch_size=2` olan PyTorch `DataLoader` nesnelerini döndüren mimari kuruldu.

### ⏳ Sıradaki Aşamalar (Bundan Sonra Yapılacaklar)

* **Aşama 2'nin Devamı: Sıfırdan Test (Zero-Shot Baseline)** (Bedirhan Öztürk)
  - Eğitimsiz SAM modeli (ViT-B) yüklenecek ve `test` splitindeki 100 resim üzerinde bölütleme denemesi yapılacak.
  - Kötü de olsa referans (baseline) olması için Dice ve IoU metrikleri hesaplanıp tabloya işlenecek.
  
* **Aşama 3: Model Mimarisi ve Eğitim Döngüsü (Fine-Tuning)** (Abdullah Aydoğan)
  - SAM "Image Encoder" (ViT) parametreleri tamamen dondurulacak (`requires_grad = False`). Sadece "Mask Decoder" eğitime açılacak.
  - Tıbbi standarda uygun **BCE Loss + Dice Loss** tanımlanacak. **AdamW** optimizer kullanılacak.
  - Epoch bazlı PyTorch eğitim döngüsü yazılacak. Tensörboard grafikleriyle validation loss takibi yapılacak.

* **Aşama 4: Model Değerlendirmesi ve Görselleştirme** (Bedirhan Öztürk)
  - Eğitilmiş model ile Aşama 2'deki 100 test resminde yeniden Inference (çıkarım) alınacak. Dice ve IoU metriklerinin sıfır modele göre ne kadar geliştiği ölçülecek.
  - Niteliksel görselleştirmeler (Qualitative Visualizations) yapılacak (Örn: Işık parlaması olan polipte orijinal maske vs. bizim modelin tahmini).
  - Hata analizi yapılarak (model nerede patladı?) rapora notlar düşülecek.

* **Aşama 5: Finalizasyon, Kod Temizliği ve Rapor** (Tüm Ekip)
  - Modüler `.py` dosyaları temizlenecek veya derli toplu bir `Final_Notebook.ipynb` oluşturulacak.
  - GitHub README dosyası sonuç grafikleriyle (Before/After) süslenecek.
  - Hocaya sunulacak final raporu ve sunum slaytları donanım dostu (T4 GPU limitleri) mühendislik çözümleri vurgulanarak hazırlanacak.
