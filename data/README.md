# Veri Setleri Rehberi ve Şema Dokümantasyonu

Ham veri dosyaları boyut ve lisans koşulları gereği GitHub deposuna eklenmez (`data/raw/` `.gitignore` ile hariç tutulmuştur). Aşağıda iki veri setinin resmi indirme bağlantıları, doğrulanan dosya yapıları, sütun/birim tabloları ve veri ayırma kuralları yer almaktadır.

---

## 1. Ana Veri Seti: Stanford Energy Control Laboratory (INR21700-M50T)

* **Yayın:** G. Pozzato, A. Allam, and S. Onori, *"Lithium-ion battery aging dataset based on electric vehicle real-driving profiles,"* *Data in Brief*, vol. 41, 107995, 2022.
* **Resmi OSF Sayfası:** `https://osf.io/qsabn/` (View-only API belirteci: `view_only=2a03b6c78ef14922a3e244f3d549de78`)
* **Resmi Dropbox Yedeği (`README.xlsx` içinden):** `https://www.dropbox.com/scl/fo/3ss0age6ggfcm67okldhw/h?rlkey=tnczvb82gukfe2n4gol2uyo7x&dl=0`
* **Hücre:** LG Chem **INR21700-M50T** (NMC811 / Grafit-SiOₓ, nominal kapasite **4.85 Ah**, çalışma gerilimi **2.50 V – 4.20 V**, test kabini ortam sıcaklığı **23 °C**).
* **Şarj C-Oranları (Hücrelere Göre):**
  * `C/4` (~1.21 A): `W4`, `W7` (141. çevrimde sonlandırıldı), `V4`
  * `C/2` (~2.43 A): `W5`, `W8`
  * `1C` (~4.85 A): `W9`, `V5`
  * `3C` (~14.55 A): `W3` (75. çevrimde sonlandırıldı), `W10`, `G1`

### 1.1. Programatik İndirme (Yerel ve Google Colab)
OSF WaterButler API üzerinden işlenmiş `.mat` dosyaları doğrudan indirilebilir (tam katalog `results/tables/stanford_osf_manifest.csv` içindedir):
* Tanı dosyaları (`diagnostic_tests/_processed_mat/`): `capacity_test.mat` (75.1 MB), `HPPC_test.mat` (17.5 MB), `EIS_test.mat` (0.05 MB), `data_analysis.m`
* Çevrim dosyaları (`cycling_tests/Cycling_1/_processed_mat/`): `W4.mat`, `W8.mat`, `W9.mat`, `W10.mat` (~96–108 MB/dosya)

### 1.2. Doğrulanmış Dosya Şemaları ve Birimler
1. **Tanı Dosyaları (`capacity_test.mat`, `HPPC_test.mat`)** — 15×10 hücre dizisi (`row_diag_number`: 1..15, `col_cell_label`: `['W3','W4','W5','W7','W8','W9','W10','G1','V4','V5']`):
   * `time` [s]: Test zamanı ($\Delta t = 1.0\text{ s}$, 1 Hz)
   * `vcell` [V]: Hücre terminal gerilimi
   * `curr` [A]: Akım (**Pozitif = Şarj, Negatif = Deşarj**; örn. `capacity_test` C/20 = `-0.243 A`, `HPPC` darbeleri `[-4.85, +3.00] A`)
   * `cap` [Ah]: Kümülatif kapasite
2. **EIS Dosyası (`EIS_test.mat`)** — `soc_level`: `[20, 50, 80]` % SOC:
   * `freq` [Hz], `re_z` [Ohm], `im_z` [Ohm]
   * *Not:* `W4`, `W5`, `W7`, `G1` hücreleri için donanım nedeniyle EIS kaydı mevcut değildir (`0` tanı).
3. **İşlenmiş Çevrim Dosyaları (`W4.mat`, `W8.mat`, `W9.mat`, `W10.mat`)**:
   * `t_full_vec_M1_NMC25degC` [s]: Toplam zaman ($\Delta t_{\text{med}} = 0.10\text{ s}$ UDDS, CC/CV/dinlenme adımlarında ~1 s)
   * `I_full_vec_M1_NMC25degC` [A]: Akım (**Pozitif = Şarj, Negatif = Deşarj**; `W10` 3C adımı `+14.57 A`)
   * `V_full_vec_M1_NMC25degC` [V]: Terminal gerilimi
   * `Step_Index_full_vec_M1_NMC25degC` [-]: Protokol adımı (`0`: 0V başlangıç artifaktı, `2–3`: ön 1C CC-CV, `7–8`: hedef C-oranında 4.0 V CC-CV, `9–10`: C/4 4.2 V CC-CV, `11`: 30 dk dinlenme, `13`: C/4 %20 deşarj, `14`: UDDS %60 deşarj)
   * `ch_cap_full_vec_M1_NMC25degC`, `dis_cap_full_vec_M1_NMC25degC` [Ah]: Şarj / deşarj kapasitesi
   * *Kritik Tespit:* İşlenmiş `.mat` dosyalarında `Aux_Temperature(°C)` yer almamaktadır; sıcaklık yalnızca ham `.xlsx` dosyalarında bulunur.

---

## 2. Yardımcı Veri Seti: NASA Li-ion Battery Aging Dataset

* **Kaynak:** B. Saha & K. Goebel, NASA Ames PCoE (`https://catalog.data.gov/dataset/li-ion-battery-aging-datasets`)
* **Doğrudan Arşiv:** `https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip` (~200 MB)
* **İncelenen Hücreler:** `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` (24 °C), `B0029.mat` (43 °C), `B0053.mat` (4 °C).
* **Yapı:** İç içe MATLAB yapısı (`cycle[i]` → `type ∈ {'charge', 'discharge', 'impedance'}`, `ambient_temperature`, `time`, `data`).
  * `charge`/`discharge`: `Time` [s], `Voltage_measured` [V], `Current_measured` [A], `Temperature_measured` [°C], `Current_charge`/`Current_load` [A], `Voltage_charge`/`Voltage_load` [V], deşarjda ayrıca `Capacity` [Ah].
  * `impedance`: `Re` [Ohm], `Rct` [Ohm], `Battery_impedance` [Ohm].
* **Kalite Bulguları:** `B0005`–`B0007` içerisinde tekil anormal gerilim sıçramaları (`0.003 V` – `8.39 V`) ve `B0018` içerisinde 6 adet `NaN` değeri tespit edilmiştir; Aşama 2'de filtrelenecektir.

---

## 3. Veri Sızıntısını Önleme ve Birleştirmeme Kuralı
* Stanford (21700, 4.85 Ah, UDDS) ve NASA (18650, ~2.0 Ah, sabit/kare dalga deşarj) hücreleri fiziksel olarak birleştirilmez.
* **Eğitim / Kalibrasyon Seti:** Stanford `W4` (C/4), `W8` (C/2), `W9` (1C), `W10` (3C) erken tanı (`Diag_1`) ve `Cycling_1` verileri.
* **Bağımsız Doğrulama Seti:** Eğitimde kullanılmayan `W5` (C/2) ve `V4` (C/4) hücrelerinin `capacity_test` / `HPPC_test` verileri ile ileri çevrim (`Diag_2+`) verileri.
