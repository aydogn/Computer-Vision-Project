# YZV416E Computer Vision - Project: SAM-Med

**Adapting Segment Anything - Decoder Fine-tuning on Domain-Specific Datasets (Medical Image Segmentation)**

**Ekip:** SAM-Med
- **Omer Faruk Satik**: Core Codebase & Data Pipeline Lead
- **Bedirhan Ozturk**: Evaluation & Inference Lead
- **Abdullah Aydogan**: Model Training & Optimization Lead

---

## Proje Ozeti

Bu proje, Meta'nin **Segment Anything Model (SAM)** mimarisini (ViT-B), endoskopik goruntulerdeki poliplerin (gastrointestinal polipler) piksel seviyesinde bolutlenmesi amaciyla **Kvasir-SEG** veri seti uzerinde ince ayar (fine-tuning) yaparak tip alanina adapte etmeyi amaclamaktadir. Donanim sinirlarina (ornegin 16GB VRAM) saygi duymak adina, SAM'in agir ViT kodlayicisi (encoder) dondurulacak ve sadece hafif maske cozucusu (decoder) egitilecektir.

---

## Kurulum ve Baslangic Yonergesi

Gorevi devralan ekip arkadasinin veya projeyi test edecek hocanin projeyi kendi bilgisayarinda ya da Colab'da sifirdan calistirabilmesi icin asagidaki adimlari sirayla uygulamasi gerekir.

### 0. Projenin Klonlanmasi

```bash
git clone https://github.com/aydogn/Computer-Vision-Project.git
cd Computer-Vision-Project
```

### 1. Conda Ortaminin Olusturulmasi ve Aktif Edilmesi

```bash
conda create -n sam-med python=3.10 -y
conda activate sam-med
```

### 2. Kutuphanelerin Kurulumu

GPU kullaniliyorsa once CUDA destekli PyTorch kurulumu onerilir:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Ardindan proje bagimliliklari kurulur:

```bash
pip install -r requirements.txt
```

`requirements.txt` icinde Meta'nin **segment-anything** reposu `git+https://...` baglantisi ile otomatik kurulur. Ek olarak NumPy ve OpenCV surumleri, mevcut PyTorch ortamlariyla uyumlu olacak sekilde sabitlenmistir.

### 3. Veri Seti ve Model Agirliklarinin Indirilmesi

Kvasir-SEG veri setini Hugging Face uzerinden indirmek icin:

```bash
python download_data.py
```

SAM ViT-B pretrained checkpoint dosyasini indirmek icin Windows PowerShell:

```powershell
mkdir checkpoints
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile checkpoints\sam_vit_b_01ec64.pth
```

Linux veya Colab:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
```

### 4. Kurulumun Test Edilmesi

Data pipeline'in dogru calistigini test etmek icin:

```bash
python test_dataset.py
```

Bu komut basarili olursa proje ana dizininde `test_output_v2.png` isimli bir ornek cikti olusturur.

### 5. Zero-Shot Baseline Calistirilmasi

Egitimsiz/pretrained SAM ViT-B modelini Kvasir-SEG test splitinde degerlendirmek icin:

```bash
python evaluate_zero_shot.py
```

Hizli smoke test icin:

```bash
python evaluate_zero_shot.py --max-images 5 --num-visualizations 5
```

Varsayilan yollar:

```text
Dataset root: kvasir-seg
Checkpoint: checkpoints/sam_vit_b_01ec64.pth
Output directory: results
```

Uretilen ciktilar:

```text
results/zero_shot_baseline.csv
results/zero_shot_summary.txt
results/zero_shot_visualizations/
```

---

## Zero-Shot Baseline Yaklasimi

Aşama 2'nin devami kapsaminda SAM modeli fine-tune edilmeden degerlendirilmistir. Bu baseline, ileride Aşama 3'te egitilecek decoder fine-tuned modelin performansini karsilastirmak icin referans olarak kullanilacaktir.

Kullanilan protokol:

- Pretrained SAM ViT-B checkpoint yuklendi.
- Kvasir-SEG `test` splitindeki 100 goruntu kullanildi.
- Her ground-truth maskeden bounding box prompt cikarildi.
- SAM `SamPredictor` ile `multimask_output=False` ayarinda tek maske tahmini alindi.
- Tahmin maskesi ile ground-truth maske arasinda Dice ve IoU metrikleri hesaplandi.
- Ilk 10 ornek icin niteliksel gorsellestirme kaydedildi.

Bu asamada model agirliklari guncellenmemistir. Fine-tuning islemi Aşama 3 kapsaminda yapilacaktir.

### Zero-Shot Sonuclari

`results/zero_shot_summary.txt` dosyasina kaydedilen sonuclar:

```text
Evaluated images: 100
Skipped images: 0
Mean Dice: 0.818568
Median Dice: 0.931399
Mean IoU: 0.759482
Median IoU: 0.871606
```

Dataset dogrulama sonucu:

```text
train: images=800 masks=800 matched=800 missing_masks=0 extra_masks=0
validation: images=100 masks=100 matched=100 missing_masks=0 extra_masks=0
test: images=100 masks=100 matched=100 missing_masks=0 extra_masks=0
```

---

## Proje Yol Haritasi ve Mevcut Durum

Projenin **5 asamali** bir uygulama plani bulunmaktadir. Mevcut durumda Aşama 1, Aşama 2 ve Aşama 2'nin devami olan zero-shot baseline tamamlanmistir.

### Tamamlanan Asamalar

* **Aşama 1: Ortam Kurulumu ve Veri Boru Hatti** (Omer Faruk Satik)
  - GitHub reposu ve Conda ortami ayarlandi.
  - Kvasir-SEG indirme scripti (`download_data.py`) yazildi.
  - SAM ViT-B checkpoint'i sisteme entegre edildi.
  - Orijinal goruntu ile maskeleri okuyup eslestiren PyTorch `Dataset` sinifi yazildi.
  - Maskeler binary (0 ve 1) tensor formatina zorlandi.

* **Aşama 2: Veri On Isleme (Data Preprocessing)** (Omer Faruk Satik)
  - `torchvision` yerine `albumentations` kutuphanesine gecilerek senkronize donusumler kodlandi.
  - Polip aspect-ratio'sunu korumak icin `LongestMaxSize(1024)` ve siyah piksellerle `PadIfNeeded` eklendi.
  - Egitim seti icin `HorizontalFlip`, `VerticalFlip`, `RandomRotate90` ve `ColorJitter` ile online veri artirimi saglandi.
  - SAM standartlarinda (`mean`, `std`) ImageNet normalizasyonu yapildi.
  - OOM hatalarini engellemek icin `batch_size=2` olan PyTorch `DataLoader` nesnelerini donduren mimari kuruldu.

* **Aşama 2'nin Devami: Sifirdan Test (Zero-Shot Baseline)** (Bedirhan Ozturk)
  - `evaluate_zero_shot.py` scripti eklendi.
  - Pretrained SAM ViT-B modeli yuklendi.
  - `test` splitindeki 100 goruntu uzerinde zero-shot inference calistirildi.
  - Ground-truth maskelerden bounding box prompt cikarildi.
  - Dice ve IoU metrikleri hesaplandi.
  - Sonuclar `results/zero_shot_baseline.csv` ve `results/zero_shot_summary.txt` dosyalarina kaydedildi.
  - 10 ornek icin `results/zero_shot_visualizations/` altinda gorsellestirme uretildi.
  - Baseline sonucu: Mean Dice `0.818568`, Mean IoU `0.759482`.

### Siradaki Asamalar

* **Aşama 3: Model Mimarisi ve Egitim Dongusu (Fine-Tuning)** (Abdullah Aydogan)
  - SAM "Image Encoder" (ViT) parametreleri tamamen dondurulacak (`requires_grad = False`).
  - Sadece "Mask Decoder" egitime acilacak.
  - Tibbi segmentasyon icin **BCE Loss + Dice Loss** tanimlanacak.
  - **AdamW** optimizer kullanilacak.
  - Epoch bazli PyTorch egitim dongusu yazilacak.
  - Validation loss takibi icin TensorBoard veya benzeri loglama kullanilacak.
  - Aşama 2 zero-shot baseline metrikleri, fine-tuning sonrasi performans artisini olcmek icin referans alinacak.

* **Aşama 4: Model Degerlendirmesi ve Gorsellestirme** (Bedirhan Ozturk)
  - Fine-tuned model ile ayni 100 test goruntusu uzerinde inference alinacak.
  - Dice ve IoU metrikleri zero-shot baseline ile karsilastirilacak.
  - Before/After gorsellestirmeleri hazirlanacak.
  - Zero-shot SAM, fine-tuned SAM ve ground-truth maskeler yan yana raporlanacak.
  - Hata analizi yapilarak modelin basarisiz oldugu ornekler belirlenecek.

* **Aşama 5: Finalizasyon, Kod Temizligi ve Rapor** (Tum Ekip)
  - Moduler `.py` dosyalari temizlenecek veya derli toplu bir `Final_Notebook.ipynb` olusturulacak.
  - GitHub README dosyasi final metrikler ve Before/After gorselleriyle guncellenecek.
  - Final raporu ve sunum slaytlari hazirlanacak.
  - Donanim dostu muhendislik cozumleri ve SAM decoder fine-tuning stratejisi vurgulanacak.

---

## Guncel Dosya ve Cikti Ozeti

Yeni veya guncellenen onemli dosyalar:

```text
evaluate_zero_shot.py
download_data.py
requirements.txt
results/zero_shot_baseline.csv
results/zero_shot_summary.txt
results/zero_shot_visualizations/
```
