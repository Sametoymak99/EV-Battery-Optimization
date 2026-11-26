# Elektrikli Araç Bataryalarında Çok Amaçlı Optimal Şarj Stratejisinin Nümerik Optimizasyonu

**Ders:** MAT353 – Nümerik Analiz (2026–2027 Güz Dönemi)  
**Bölüm:** Bilgisayar Mühendisliği – İstanbul Medeniyet Üniversitesi  
**Öğretim Üyesi:** Doç. Dr. Murat GEZER  
**Öğrenci:** Samet Oymak (24120205042)  
**Proje Alanı:** Elektrikli Araçlar, Batarya Sistemleri, Nümerik Optimizasyon ve Enerji Verimliliği  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sametoymak99/EV-Battery-Optimization/blob/main/notebooks/EV_Battery_Optimization_Colab_Demo.ipynb)

## Proje Özeti
Bu proje, elektrikli araçlarda kullanılan lityum iyon bataryaların (LG Chem INR21700-M50T) şarj sürecini birinci dereceden Thevenin eşdeğer devresi, birleştirilmiş (lumped) termal model ve ampirik yaşlanma göstergesi ile matematiksel olarak modellemeyi; şarj süresi, direnç kaynaklı enerji kaybı ve göreli yaşlanmayı birlikte en aza indiren akım profillerini nümerik optimizasyon yöntemleriyle (Gradient Descent, Adam, SLSQP) belirlemeyi amaçlar.

Çalışma akışı:
`Gerçek Veri → Matematiksel Model → Parametre Tahmini → Batarya Simülasyonu → Nümerik Optimizasyon → Pareto & Monte Carlo Analizi → CC-CV Karşılaştırması`

## Klasör Yapısı ve Görevleri

```text
EV-Battery-Optimization/
│
├── README.md                  # Proje özeti, kurulum ve kullanım kılavuzu
├── requirements.txt           # Sabitlenmiş Python kütüphane bağımlılıkları
├── .gitignore                 # Ham veri ve geçici dosyaları hariç tutma kuralları
│
├── docs/                      # Akademik ve teknik proje dokümantasyonu
│   ├── project_proposal.md    # Proje önerisi, kapsam ve 12 aşamalı yol haritası
│   ├── mathematical_guide.md  # ODE sistemleri, amaç fonksiyonları ve karmaşıklık türetmeleri
│   ├── methodology.md         # Veri ayrımı, sayısal çözüm ve optimizasyon metodolojisi
│   └── references.md          # Doğrulanmış IEEE formatındaki bilimsel kaynaklar
│
├── data/                      # Veri katmanı (ham veriler depoya eklenmez)
│   ├── README.md              # Stanford ve NASA veri setleri indirme, şema ve lisans bilgisi
│   ├── raw/                   # İndirilen ham Stanford (.mat/.xlsx/.csv) ve NASA (.mat) dosyaları
│   └── processed/             # Temizlenmiş döngüler, OCV tabloları ve RPT özetleri
│
├── notebooks/                 # Aşama bazlı Jupyter / Google Colab çalışma defterleri
│   └── 01_data_exploration.ipynb ... 10_monte_carlo.ipynb
│
├── src/                       # Modüler ve test edilebilir Python çekirdek kütüphanesi
│   ├── data_loader.py         # Stanford ve NASA dosyalarını okuma ve şema denetimi
│   ├── preprocessing.py       # Veri temizleme, döngü ayrıştırma ve SOC hesabı
│   ├── battery_model.py       # Thevenin + termal + yaşlanma ODE'leri ve çözücüler (Euler/RK4/solve_ivp)
│   ├── parameter_estimation.py# OCV, R0, R1, C1, Rth, Cth parametre kestirimi
│   ├── objective_functions.py # Normalize J1 (süre), J2 (enerji kaybı), J3 (yaşlanma)
│   ├── constraints.py         # Gerilim, sıcaklık, akım ve SOC kısıt denetimleri
│   ├── cc_cv.py               # Referans geleneksel CC-CV şarj simülatörü
│   ├── optimizers.py          # Gradient Descent, Adam ve SciPy SLSQP uygulamaları
│   └── evaluation.py          # Pareto analizi, Monte Carlo simülasyonu ve grafik üretimi
│
├── tests/                     # Birim testleri ve doğrulama senaryoları (pytest)
│   └── test_data_loader.py
│
├── results/                   # Üretilen çıktılar
│   ├── figures/               # Birimli grafikler (PNG/PDF)
│   ├── tables/                # Şema özetleri, performans ve hata tabloları (CSV)
│   └── simulation_outputs/    # Tekrar üretilebilir simülasyon sonuçları
│
├── presentation/              # Akademik sunum materyalleri (en fazla 10 slayt, 15 dk)
│   ├── slides/
│   └── figures/
│
└── report/                    # 8–15 sayfalık MAT353 final proje raporu
    ├── final_report.md
    └── references.bib
```

## Kurulum ve Çalıştırma
**Google Colab'da Tek Tıkla Çalıştırma:** [EV_Battery_Optimization_Colab_Demo.ipynb](https://colab.research.google.com/github/Sametoymak99/EV-Battery-Optimization/blob/main/notebooks/EV_Battery_Optimization_Colab_Demo.ipynb)

Yerel kurulum için:
```bash
git clone https://github.com/Sametoymak99/EV-Battery-Optimization.git
cd EV-Battery-Optimization
pip install -r requirements.txt
pytest tests/ -v
```

Tüm aşama defterlerini (`01..10`) Google Colab üzerinde ham veri ile çalıştırmak için `data/README.md` içerisindeki indirme adımlarını uygulayabilirsiniz; `EV_Battery_Optimization_Colab_Demo.ipynb` ise ek kurulum gerektirmeden doğrudan çalışır.

## Veri Seti ve Lisans Notu
* **Ana Veri Seti:** Stanford Energy Control Laboratory – INR21700-M50T Veri Seti (`https://osf.io/qsabn/`) [1].
* **Yardımcı Veri Seti:** NASA Ames PCoE Li-ion Battery Aging Dataset (`https://catalog.data.gov/dataset/li-ion-battery-aging-datasets`) [2].
* Ham veri dosyaları boyut ve lisans kuralları gereği doğrudan git deposuna eklenmez; indirme adresleri ve doğrulama adımları `data/README.md` dosyasında yer almaktadır.
