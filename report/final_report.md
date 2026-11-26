# 1. Kapak

**İSTANBUL MEDENİYET ÜNİVERSİTESİ**  
**MÜHENDİSLİK VE DOĞA BİLİMLERİ FAKÜLTESİ**  
**BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ**

---

**Ders:** MAT353 – Nümerik Analiz  
**Dönem:** 2026–2027 Güz Dönemi  
**Öğretim Üyesi:** Doç. Dr. Murat GEZER  
**Proje Başlığı:** Elektrikli Araç Bataryalarında Çok Amaçlı Optimal Şarj Stratejisinin Nümerik Optimizasyonu  
**Proje Türü:** Bireysel Dönem Projesi  
**Öğrenci:** Samet Oymak (24120205042)  
**Teslim Tarihi:** 13 Aralık 2026  

---

# 2. Özet

Elektrikli araçlarda lityum iyon bataryaların hızlı şarj edilmesi, kullanıcı beklentileri açısından kritik önem taşımakla birlikte yüksek akımların yol açtığı Joule enerji kayıplarını, ısınmayı ve hücre yaşlanmasını hızlandırmaktadır. Bu projede, LG Chem INR21700-M50T batarya hücresinin şarj süreci birinci dereceden Thevenin eşdeğer devresi, birleştirilmiş termal model ve ampirik göreli yaşlanma göstergesiyle modellenmiş; şarj süresi, enerji kaybı ve göreli yaşlanmayı birlikte minimize eden optimal akım profilleri nümerik yöntemlerle belirlenmiştir. Model parametreleri Stanford Energy Control Laboratory deney verilerinin kalibrasyon bölümünden doğrusal olmayan en küçük kareler yöntemiyle kestirilmiş, elektrokimyasal empedans spektroskopisi ve görülmemiş bağımsız doğrulama hücreleri üzerinde test edilerek veri sızıntısı önlenmiştir. NASA Ames batarya yaşlanma veri seti ise ayrı bir fiziksel referans olarak sıcaklık ve kapasite-empedans eğilimlerinin nitel kontrolünde kullanılmıştır. Diferansiyel denklem çözümünde İleri Euler, dördüncü derece Runge-Kutta ve SciPy uyarlanabilir çözücüsü yakınsama mertebesi, kararlılık sınırı ve çalışma süresi bakımından kıyaslanmıştır. Çok amaçlı optimizasyon problemi için SOC-dilimli kontrol parametrizasyonu altında projeksiyonlu ve cezalı Gradient Descent, Adam ve SciPy SLSQP algoritmaları geleneksel sabit akım–sabit gerilim stratejisiyle karşılaştırılmıştır. Dengeli ağırlık senaryosunda optimize edilen stratejiler enerji kaybını yaklaşık yüzde on iki ila on dört, göreli yaşlanma göstergesini ise yüzde dokuz buçuk ila on bir oranında azaltmıştır. Son olarak yirmi ağırlık kombinasyonuyla Pareto cephesi çıkarılmış ve iki yüz örneklemli Monte Carlo analiziyle parametre belirsizliklerinin kısıt güvenilirliği üzerindeki etkisi nicel olarak ortaya konmuştur.

---

# 3. Giriş

Elektrikli araçların (EA) yaygınlaşması, yüksek enerji yoğunluğuna ve uzun çevrim ömrüne sahip lityum iyon bataryaların güvenli ve hızlı biçimde şarj edilmesini temel bir mühendislik problemine dönüştürmüştür [1], [7]. Geleneksel şarj uygulamalarında yaygın olarak kullanılan Sabit Akım–Sabit Gerilim (CC-CV) protokolü, uygulanması kolay olmakla birlikte enerji verimliliği ve hücre yaşlanması arasında sistematik bir ödünleşim mekanizması sunmaz. Şarj akımının artırılması toplam şarj süresini kısaltırken, iç direnç üzerinden açığa çıkan $I^2 R$ ısı kayıplarını karesel olarak büyütmekte, hücre sıcaklığını yükseltmekte ve elektrot-elektrolit arayüzeyinde bozunmayı tetiklemektedir [1], [7]. Dolayısıyla EA batarya yönetim sistemlerinde yalnızca şarj süresini değil, aynı anda direnç kaynaklı enerji kaybını ve elektro-termal yaşlanmayı da gözeterek gerilim ve sıcaklık sınırlarını ihlal etmeyen çok amaçlı optimal şarj profillerinin geliştirilmesi gerekmektedir.

Literatürde batarya modellemesi ve optimal şarj kontrolü farklı yaklaşımlarla ele alınmaktadır. Pozzato, Allam ve Onori [1], gerçek elektrikli araç sürüş profilleri (UDDS) ve farklı şarj C-oranları (C/4, C/2, 1C, 3C) altında INR21700-M50T hücrelerinin kapasite, HPPC ve EIS yaşlanma verilerini yayımlayarak veri temelli modelleme çalışmalarının önünü açmıştır. Saha ve Goebel [2] ise NASA Ames Prognostics veri deposunda 18650 hücreler için farklı ortam sıcaklıklarında çevrim ömrü ve empedans ($R_e, R_{ct}$) verilerini sunmuştur. Modelleme tarafında Plett [7], birinci dereceden Thevenin eşdeğer devre modellerinin hem düşük hesaplama maliyeti hem de yüksek gerilim kestirim doğruluğu sayesinde gömülü optimizasyon ve kontrol için elverişli olduğunu göstermektedir. Optimizasyon tarafında ise gradyan tabanlı birinci derece yöntemler (Gradient Descent ve moment uyarlamalı Adam [4]) ile doğrudan kısıtlı sıralı karesel programlama (SLSQP [5], [6]) öne çıkmaktadır.

Birinci derece ceza tabanlı yöntemler (GD, Adam) kodlanması esnek ve ölçeklenebilir olmasına karşın ceza katsayısına ve öğrenme oranına duyarlıdır; buna karşın SLSQP gibi aktif küme temelli yöntemler düzgün doğrusal olmayan eşitsizlik kısıtlarını (terminal gerilimi ve sıcaklık üst sınırları) doğrudan Hessian yaklaşımıyla çözerek az iterasyonda yüksek doğruluk sunabilmektedir [5], [6]. Bu çalışmada, söz konusu algoritmalar aynı gerçek deney verisinden kalibre edilmiş fiziksel batarya modeli üzerinde adil koşullarda karşılaştırılmış; ek olarak Nümerik Analiz dersi kapsamında ODE çözücülerinin yakınsama ve kararlılık davranışları, sonlu fark gradyan hassasiyeti, Pareto cephesi ve Monte Carlo belirsizlik yayılımı bütüncül biçimde incelenmiştir.

---

# 4. Yöntem

Bu bölümde batarya sisteminin durum uzayı modeli, kısıtlar, çok amaçlı optimizasyon formülasyonu, sayısal entegrasyon ve türev yöntemleri ile algoritmaların karmaşıklık analizi sunulmaktadır.

### 4.1. Durum Uzayı ve Fiziksel Denklemler
Batarya dinamiği, birinci dereceden Thevenin eşdeğer devresi, birleştirilmiş (lumped) termal denklem ve ampirik yaşlanma göstergesi ile dört boyutlu durum vektörü $x(t) = [\text{SOC}(t), V_p(t), T(t), D(t)]^T$ üzerinden modellenmiştir [7]. Kontrol değişkeni şarj akımı $u(t) = I(t)\;[\text{A}]$ (şarj yönünde pozitif) olarak tanımlanmıştır:

1. **Doluluk Oranı (SOC) Dinamiği:**
   $$\frac{d\text{SOC}}{dt} = \frac{\eta\, I(t)}{3600\, Q}$$
   Burada $\eta$ Coulombik verimlilik, $Q\;[\text{Ah}]$ hücre kapasitesi ve $3600$ saat–saniye dönüşüm katsayısıdır. Optimizasyonda $\text{SOC}(0) = 0.20$ başlangıcından $\text{SOC}(t_f) = 0.90$ hedefine ulaşılmasını denetler.

2. **Polarizasyon Gerilimi ($V_p$):**
   $$\frac{dV_p}{dt} = -\frac{V_p(t)}{R_1 C_1} + \frac{I(t)}{C_1}$$
   $R_1\;[\Omega]$ polarizasyon direnci ve $C_1\;[\text{F}]$ polarizasyon kapasitesi olup $\tau_1 = R_1 C_1\;[\text{s}]$ gevşeme zaman sabitini belirler.

3. **Terminal Gerilimi ($V_t$):**
   $$V_t(t) = \text{OCV}(\text{SOC}(t)) + I(t) R_0 + V_p(t)$$
   $\text{OCV}(\text{SOC})$ C/20 tanı eğrisinden şekil-korumalı PCHIP interpolasyonu ile elde edilmiştir; $R_0\;[\Omega]$ ohmik iç dirençtir. Güvenlik kısıtı $V_t(t) \le V_{\max} = 4.20\text{ V}$ biçimindedir.

4. **Birleştirilmiş (Lumped) Termal Model:**
   $$C_{\text{th}} \frac{dT}{dt} = I(t)^2 R_0 + \frac{V_p(t)^2}{R_1} - \frac{T(t) - T_{\text{amb}}}{R_{\text{th}}}$$
   $C_{\text{th}}\;[\text{J/K}]$ ısıl sığa, $R_{\text{th}}\;[\text{K/W}]$ ısıl direnç ve $T_{\text{amb}} = 23.0\ ^\circ\text{C}$ ortam sıcaklığıdır. Entropik ısı ve hücre içi radyal sıcaklık gradyanı başlangıç modelinde ihmal edilmiştir. Kısıt $T(t) \le T_{\max} = 45.0\ ^\circ\text{C}$ olarak uygulanmıştır.

5. **Ampirik Göreli Yaşlanma Göstergesi ($D$):**
   $$\frac{dD}{dt} = k_0 \exp\!\left(-\frac{E_a}{R_g (T(t)+273.15)}\right) \left(\frac{|I(t)|}{Q}\right)^p \left(1 + \alpha_{\text{soc}}\,\text{SOC}(t)^2\right)$$
   Burada $k_0 = 0.015\text{ s}^{-1}$, $E_a = 31500\text{ J/mol}$, $R_g = 8.3145\text{ J/(mol}\cdot\text{K)}$, $p = 1.5$ ve $\alpha_{\text{soc}} = 0.5$ literatür temelli göreli stres katsayılarıdır. $D(t)$ mutlak kapasite kaybını (SOH) doğrudan ölçmez; farklı stratejilerin elektro-termal stresini kıyaslayan **göreli yaşlanma indeksi** olarak kullanılır.

6. **Direnç Kaynaklı Enerji Kaybı:**
   $$E_{\text{loss}} = \int_0^{t_f} \left(I(t)^2 R_0 + \frac{V_p(t)^2}{R_1}\right) dt \quad [\text{J}]$$

### 4.2. Çok Amaçlı Optimizasyon ve SOC-Dilimli Parametrizasyon
Amaç fonksiyonları, 1.0C geleneksel CC-CV stratejisinden elde edilen $(t_{\text{ref}}, E_{\text{ref}}, D_{\text{ref}})$ ölçekleriyle boyutsuzlaştırılmıştır:
$$J(\mathbf{u}) = w_1 \frac{t_f(\mathbf{u})}{t_{\text{ref}}} + w_2 \frac{E_{\text{loss}}(\mathbf{u})}{E_{\text{ref}}} + w_3 \frac{D(t_f) - D(0)}{D_{\text{ref}}}, \qquad \sum_{i=1}^3 w_i = 1,\; w_i \ge 0$$

Serbest bitiş zamanlı ($t_f$) problemde sabit zaman adımlı ayrıklaştırma $\text{SOC}(t_f)=0.90$ koşulunda türev süreksizliğine yol açtığından, $[0.20, 0.90]$ aralığı $N=10$ eşit $\Delta\text{SOC}_k = 0.07$ dilimine bölünmüş ve karar vektörü $\mathbf{u} = [I_1, \dots, I_{10}]^T \in [0.50, 9.72]\text{ A}$ seçilmiştir. Her dilimin süresi analitik olarak $\Delta t_k = \frac{3600\, Q\, \Delta\text{SOC}_k}{\eta I_k}$ biçiminde hesaplanarak hem $t_f(\mathbf{u})$ fonksiyonunun $C^\infty$ türevlenebilirliği hem de $\text{SOC}(t_f) = 0.90$ hedefinin tam sağlanması garanti edilmiştir.

```mermaid
flowchart LR
    A["Gerçek Veri<br/>(Stanford & NASA)"] --> B["Temizleme & Ayrıştırma<br/>(Step 0, NaN, Coulomb SOC)"]
    B --> C["Parametre Tahmini<br/>(HPPC LSQ, EIS, W10 Termal)"]
    C --> D["Bağımsız Doğrulama<br/>(W5, V4 RMSE Kontrolü)"]
    D --> E["ODE Çözücü Analizi<br/>(Euler, RK4, solve_ivp)"]
    E --> F["CC-CV & Optimizasyon<br/>(GD, Adam, SLSQP)"]
    F --> G["Pareto & Monte Carlo<br/>(20 Ağırlık, N=200 Seed=2026)"]
```

### 4.3. Sayısal Yöntemler, Gradyanlar ve Big-O Karmaşıklığı
ODE sistemi artırılmış 5 durumlu vektör $\mathbf{y} = [\text{SOC}, V_p, T, D, E_{\text{loss}}]^T$ üzerinde İleri Euler ($\mathcal{O}(h)$), klasik 4. derece Runge-Kutta (RK4, $\mathcal{O}(h^4)$) ve SciPy `solve_ivp` (RK45) ile çözülmüştür [6]. Açık Euler yöntemi için doğrusal kararlılık adım sınırı $\Delta t_{\text{sınır}} = 2\min(R_1 C_1, R_{\text{th}} C_{\text{th}})$ olarak türetilmiştir. Gradient Descent ve Adam uygulamalarında kutu sınırları için Öklidyen projeksiyon $\Pi_{[I_{\min}, I_{\max}]}(\cdot)$, gerilim ve sıcaklık kısıtları için ise 2 kademeli karesel dış ceza ($\rho_V: 500 \to 5000$, $\rho_T: 10 \to 100$) ve ileri sonlu fark gradyanı ($h = 10^{-4}\text{ A}$) kullanılmıştır. SLSQP [5] ise aşama bazlı düzgün marj vektörlerini ($V_{\max} - \max V_{t,k} \ge 0$, $T_{\max} - \max T_k \ge 0$) doğrudan eşitsizlik kısıtı olarak çözer.

| İşlem / Algoritma | Tek İterasyon Maliyeti | Toplam Karmaşıklık (Big-O) | Açıklama |
| :--- | :---: | :---: | :--- |
| Tek Batarya Simülasyonu (RK4) | $\mathcal{O}(M)$ | $\mathcal{O}(M)$ | $M$: Toplam zaman adımı sayısı |
| Sonlu Fark Gradyan Hesabı | $\mathcal{O}(N \cdot M)$ | $\mathcal{O}(N \cdot M)$ | İleri farkta $N+1$, merkezi farkta $2N$ simülasyon |
| Gradient Descent (Armijo LS) | $\mathcal{O}((N + L)\cdot M)$ | $\mathcal{O}(K_{\text{GD}} (N + L) M)$ | $L$: Geri izleme adım sayısı |
| Adam Optimizer [4] | $\mathcal{O}(N \cdot M)$ | $\mathcal{O}(K_{\text{Adam}} N M)$ | Sabit $N+1$ simülasyon/iterasyon |
| SciPy SLSQP [5], [6] | $\mathcal{O}(N \cdot M + N^3)$ | $\mathcal{O}(K_{\text{SLSQP}} (N M + N^3))$ | QP alt problemi $\mathcal{O}(N^3)$, $N=10$ için ihmal edilebilir |
| Monte Carlo ($S$ örneklem) | $\mathcal{O}(M)$ | $\mathcal{O}(S \cdot M)$ | $S=200$ bağımsız simülasyon |

**Teknoloji Seçimi:** Tüm sayısal hesaplamalar Python 3.12 ortamında vektörel `NumPy` dizileri ve `SciPy` (`least_squares`, `solve_ivp`, `PchipInterpolator`, `minimize(SLSQP)`) kütüphaneleri ile gerçekleştirilmiş; durum vektörü küçük ($d=5$) ve karar boyutu ($N=10$) olduğundan düşük ek yük nedeniyle CPU üzerinde milisaniye düzeyinde yüksek performans elde edilmiştir.

---

# 5. Uygulama

### 5.1. Veri Setlerinin İncelenmesi ve Ön İşleme
Ana veri seti olarak Stanford Energy Control Laboratory'nin INR21700-M50T veri seti [1] kullanılmıştır. WaterButler API üzerinden tanı dosyaları (`capacity_test.mat`, `HPPC_test.mat`, `EIS_test.mat`), dört C-oranını temsil eden `Cycling_1` dosyaları (`W4.mat`: C/4, `W8.mat`: C/2, `W9.mat`: 1C, `W10.mat`: 3C) ve işlenmiş `.mat` dosyalarında bulunmayan hücre yüzey sıcaklığını (`Aux_Temperature`) almak amacıyla ham Arbin dosyası (`INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N25_W10_Channel_6.2.xlsx`, 186,198 satır) indirilmiştir. Gerçek `.mat` verilerinin incelenmesiyle şarj akımının pozitif, deşarj akımının negatif işaretli olduğu ve her `Cycling_1` dosyasının `Step 0` ilk satırında `0.0 V` birleştirme artifaktı içerdiği tespit edilmiştir.

Temizleme aşamasında `W4`, `W8`, `W9` ve `W10` dosyalarından `Step 0 / 0 V` artifaktları çıkarılmış ve her biri 24 yaşlanma döngüsüne ayrıştırılmıştır. `Diag_1` C/20 tanı testlerinde kayıtlı kapasite ile yamuk Coulomb sayımıyla hesaplanan SOC arası maksimum mutlak fark tüm kalibrasyon/doğrulama hücrelerinde `0.0` (tam uyum) olarak doğrulanmıştır. Yardımcı veri seti olarak NASA Ames PCoE 18650 setinden `B0005`, `B0006`, `B0007`, `B0018` (24 °C), `B0029` (43 °C) ve `B0053` (4 °C) incelenmiş; `B0005`–`B0007` içindeki tekil gerilim sıçramaları (`0.003–8.39 V`) ve `B0018` içindeki 6 adet `NaN` temizlenmiştir. Farklı hücre kimyası (~2.0 Ah 18650 vs. 4.85 Ah 21700) nedeniyle NASA verileri Stanford verisiyle birleştirilmemiş, yalnızca sıcaklık ve yaşlanma eğiliminin bağımsız kontrolünde kullanılmıştır.

### 5.2. Parametre Kestirimi ve Veri Sızıntısız Doğrulama
Veri sızıntısını önlemek için `W4, W8, W9, W10` (`Diag_1`) kalibrasyon seti, `W5` (`Diag_2`) ve `V4` (`Diag_1`) ise kalibrasyona hiç dahil edilmeyen **bağımsız doğrulama seti** olarak ayrılmıştır. `capacity_test.mat` üzerinden ortalama kapasite $Q = 4.8599 \pm 0.0254\text{ Ah}$ ve ortalama PCHIP `OCV(SOC)` eğrisi elde edilmiştir. `HPPC_test.mat` (`0.15 ≤ SOC ≤ 0.95`) üzerinde `least_squares` ile $R_0 = 0.02865 \pm 0.00369\ \Omega$, $R_1 = 0.01585 \pm 0.00038\ \Omega$ ve $C_1 = 3795.2 \pm 128.8\text{ F}$ ($\tau_1 \approx 60.2\text{ s}$) kestirilmiştir. `EIS_test.mat` (`Im(Z) ≈ 0`) ölçümlerinden elde edilen $R_{0,\text{EIS}} = 0.02555 \pm 0.00057\ \Omega$ değeri `W8, W9, W10` hücrelerinin HPPC direnciyle (`0.0265–0.0271 Ω`) yüksek uyum göstermiştir. `W10_Channel_6.2` 3C ısıtma ve soğuma eğrisinden ($22.59 \to 33.63\ ^\circ\text{C}$) $R_{\text{th}} = 4.2547\text{ K/W}$ ve $C_{\text{th}} = 134.86\text{ J/K}$ ($\text{RMSE} = 0.1621\ ^\circ\text{C}$, $R^2 = 0.9947$) kestirilmiştir.

### 5.3. Nümerik Çözücü ve Adım Boyu Tercihleri
`DOP853` (`rtol=1e-12`) referansına karşı yapılan deneylerde İleri Euler ampirik yakınsama mertebesi `1.046`, RK4 ampirik yakınsama mertebesi ise `4.083` olarak ölçülmüştür. Açık Euler kararlılık sınırı $\Delta t_{\text{sınır}} = 120.30\text{ s}$ olarak hesaplanmış; $\Delta t = 130\text{ s}$ adımında Euler `0.5964 V` salınımlı hatayla kararsızlaşırken RK4 `0.0381 V` ile kararlı kalmıştır. Sonlu fark gradyan analizinde merkezi farkın optimum adım boyu $h^* = 10^{-5}\text{ A}$ noktasında $\nabla t_f$ bağıl hatası `4.12e-11` elde edilmiştir. Optimizasyon iterasyonlarında hız ve doğruluk dengesi için RK4 (`dt_opt_s = 10 s`, Pareto'da `20 s`), tüm nihai sonuçların raporlanmasında ise `dt_max_s = 1.0 s` yüksek çözünürlüklü doğrulama kullanılmıştır.

---

# 6. Deneysel Sonuçlar

### 6.1. Model Kalibrasyon ve Bağımsız Doğrulama Sonuçları
Tablo 1'de kalibre edilen modelin kalibrasyon hücreleri, hiç görülmemiş bağımsız doğrulama hücreleri (`W5`, `V4`) ve `Cycling_1` ileri şarj profilleri üzerindeki tahmin hataları özetlenmiştir. Şekil 1 ve Şekil 2 sırasıyla HPPC elektriksel doğrulamasını ve `W10` termal kalibrasyon uyumunu göstermektedir.

| Veri / Ayrım | Hücre | RMSE [mV / °C] | MAE [mV / °C] | $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| `HPPC_Diag1` (Kalibrasyon) | `W4 / W8 / W9 / W10` | `24.2 / 18.0 / 17.6 / 18.9` mV | `22.4 / 16.9 / 16.5 / 17.7` mV | `0.988 – 0.993` |
| `HPPC_Diag2` (**Bağımsız Doğrulama**) | **`W5` (C/2)** | **`15.69` mV** | `14.67` mV | **`0.9950`** |
| `HPPC_Diag1` (**Bağımsız Doğrulama**) | **`V4` (C/4)** | **`19.32` mV** | `18.00` mV | **`0.9920`** |
| `W10_Channel_6.2` (Termal Kalibrasyon) | `W10` (3C) | **`0.1621` °C** | `0.1248` °C | **`0.9947`** |
| `Cycling1_Cycle2_Charge` (İleri Protokol) | `W4 / W8 / W9 / W10` | `85.4 / 56.2 / 57.0 / 56.8` mV | `80.8 / 54.5 / 55.1 / 54.4` mV | `0.640 – 0.879` |

Bağımsız doğrulama hücreleri `W5` ve `V4` üzerinde sırasıyla `15.69 mV` ve `19.32 mV` RMSE değerleri elde edilmesi, kestirilen 1-RC parametrelerinin hücreler arası genellenebilirliğini kanıtlamaktadır. `Cycling_1` şarj profillerindeki `~56–85 mV` sapma ise `OCV(SOC)` eğrisinin C/20 deşarj testinden alınmasından kaynaklanan şarj–deşarj OCV histerezisini yansıtmaktadır (`Şekil 3`).

*İlgili şekiller:* [fig04_hppc_calibration_validation.png](../results/figures/fig04_hppc_calibration_validation.png), [fig04_thermal_calibration_w10.png](../results/figures/fig04_thermal_calibration_w10.png), [fig04_cycling_charge_validation.png](../results/figures/fig04_cycling_charge_validation.png).

### 6.2. ODE Çözücü ve CC-CV Referans Performansı
Şekil 4'te Euler ($\mathcal{O}(h)$) ve RK4 ($\mathcal{O}(h^4)$) yakınsama eğrileri ile $\Delta t = 130\text{ s}$ kararlılık testi görülmektedir. $\Delta t = 5\text{ s}$ adımında RK4 yalnızca `30.1 ms` içinde `2.09e-8 V` hata üretirken, İleri Euler `2.11e-3 V` hatada kalmıştır. Geleneksel CC-CV stratejisinin (`SOC: 0.20 → 0.90`, $V_{\max} = 4.20\text{ V}$) C-oranına göre sonuçları Tablo 2'de ve [fig06_cc_cv_reference.png](../results/figures/fig06_cc_cv_reference.png) grafiğinde sunulmuştur:

| C-Oranı | $I_{\text{CC}}$ [A] | CC / CV Süresi [dk] | Toplam Süre $t_f$ [dk] | Enerji Kaybı [Wh] | Göreli Yaşlanma $D$ [-] | $T_{\max}$ [°C] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.5C` | 2.43 | 84.17 / 0.00 | 84.17 | 0.3662 | `9.11e-5` | 24.12 |
| `0.7C` (Datasheet sınırı) | 3.40 | 55.75 / 4.89 | 60.64 | 0.5083 | `1.11e-4` | 25.18 |
| **`1.0C` (Referans)** | **4.86** | **35.23 / 9.58** | **44.81** (`2688.9 s`) | **0.6997** (`2518.9 J`) | **`1.388e-4`** | **27.35** |
| `1.5C` | 7.29 | 19.13 / 14.98 | 34.11 | 0.9732 | `1.82e-4` | 31.69 |
| `2.0C` | 9.72 | 11.27 / 18.62 | 29.89 | 1.1873 | `2.22e-4` | 35.67 |

C-oranı `0.5C`'den `2.0C`'ye çıktığında şarj süresi `%64.5` kısalmakta, buna karşılık enerji kaybı `3.24` katına ve göreli yaşlanma göstergesi `2.44` katına çıkmaktadır.

### 6.3. Optimizasyon Algoritmalarının Karşılaştırması ($w = [0.4, 0.2, 0.4]$)
Tablo 3, dengeli ağırlık vektörü $w = [0.4, 0.2, 0.4]$ için Gradient Descent, Adam ve SLSQP algoritmalarını 1.0C CC-CV referansıyla kıyaslamaktadır ([fig09_optimizer_comparison.png](../results/figures/fig09_optimizer_comparison.png)).

| Strateji | $t_f$ [dk] | $E_{\text{loss}}$ [Wh] | $D$ [-] | $J_1$ | $J_2$ | $J_3$ | $J_{\text{toplam}}$ | $V_{\max}$ [V] | $T_{\max}$ [°C] | Uygun? | İter / Eval | Süre [s] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CC-CV (1.0C Ref)` | 44.81 | 0.700 | `1.388e-4` | 1.000 | 1.000 | 1.000 | 1.0000 | 4.2000 | 27.35 | Evet | 0 / 1 | 0.49 |
| `Gradient Descent` | 51.82 | 0.609 | `1.246e-4` | 1.156 | 0.870 | 0.898 | 0.9957 | 4.2001 | 26.25 | Evet | 100 / 1530 | 33.72 |
| `Adam` | 52.12 | **0.600** | **`1.239e-4`** | 1.163 | **0.858** | **0.892** | **0.9937** | 4.2008 | **26.13** | Evet | 90 / 1080 | 22.57 |
| `SLSQP` | **51.11** | 0.615 | `1.257e-4` | **1.140** | 0.879 | 0.905 | 0.9941 | **4.2000** | 26.21 | Evet | **7 / 97** | **1.97** |

Çoklu başlangıç koşulu duyarlılığında (`seed = 42, 123, 2026`) her üç algoritma da `3/3` oranında tüm kısıtları sağlamıştır: **SLSQP** `J = 0.99288 ± 1.96e-7` (`9.02 s`), **Adam** `J = 0.99283 ± 3.81e-5` (`22.73 s`), **Gradient Descent** `J = 0.99627 ± 1.65e-3` (`33.04 s`).

### 6.4. Pareto Analizi ve Monte Carlo Belirsizlik Sonuçları
20 farklı ağırlık kombinasyonuyla yapılan taramada **20/20 uygun ve 20/20 Pareto-optimal** çözüm elde edilmiştir ([fig10_pareto_analysis.png](../results/figures/fig10_pareto_analysis.png)). Hız odaklı uçta (`w=[0.90, 0.05, 0.05]`) şarj süresi `32.14 dk` (`Eloss=1.141 Wh`, `D=2.12e-4`), dengeli orta noktada (`w=[0.40, 0.20, 0.40]`) `50.26 dk` (`Eloss=0.628 Wh`, `D=1.27e-4`), verimlilik odaklı uçta (`w=[0.05, 0.90, 0.05]`) ise `193.61 dk` (`Eloss=0.160 Wh`, `D=0.58e-4`) olarak gerçekleşmiştir.

Sabit `seed = 2026` ile $N_{\text{MC}} = 200$ bağımsız parametre örneklemi üzerinde gerçekleştirilen Monte Carlo analizi ([fig11_monte_carlo_uncertainty.png](../results/figures/fig11_monte_carlo_uncertainty.png)) Tablo 4'te özetlenmiştir:

| Strateji ($N=200$, `seed=2026`) | $t_f$ [dk] | $E_{\text{loss}}$ [Wh] | $D \times 10^4$ [-] | $V_{\max}$ [V] | $T_{\max}$ [°C] | $V_{\max}$ Aşım (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CC-CV (1.0C)` | `44.89 ± 0.84` | `0.700 ± 0.051` | `1.386 ± 0.025` | `4.200 ± 0.000` | `27.31 ± 0.72` | **%0.0** |
| `SLSQP Optimal (Açık Döngü)` | `51.13 ± 0.29` | **`0.616 ± 0.052`** | **`1.257 ± 0.025`** | `4.202 ± 0.012` | **`26.20 ± 0.49`** | **%45.0** |

Pearson duyarlılık analizinde $R_0$'ın enerji kaybı (`r = 0.9925`) ve maksimum gerilim (`r = 0.9906`) üzerinde, $Q$'nun şarj süresi (`r = 1.0000`) üzerinde, $R_{\text{th}}$'nin ise maksimum sıcaklık (`r = 0.8287`) ve yaşlanma (`r = 0.8299`) üzerinde baskın belirsizlik kaynağı olduğu belirlenmiştir.

---

# 7. Tartışma

* **Algoritma Performansı:** Üç algoritma arasında **SLSQP**, ikinci mertebeden kısıtlı QP yaklaşımı sayesinde GD ve Adam'a göre ~10–15 kat daha az fonksiyon değerlendirmesi (`97` vs. `1080–1530`) ve ~3–17 kat daha kısa sürede (`1.97 s` vs. `22.6–33.7 s`) başlangıç koşulundan neredeyse bağımsız (`std = 1.96e-7`) çözüm üretmiştir. Adam, kosinüs öğrenme oranı azalımıyla en düşük çoklu-seed ortalamasına (`0.99283`) ulaşmış ancak ceza katsayısına bağlı ayar gerektirmiştir. GD ise adım arama güvenlik ağına rağmen vadi boyunca zikzak yaparak en yüksek varyansı sergilemiştir.
* **Şarj Süresi – Enerji Kaybı – Yaşlanma İlişkisi:** Pareto analizi, hızlı şarj ile batarya sağlığı arasında keskin bir dışbükey ödünleşim bulunduğunu göstermektedir. Şarj süresi `45 dk` altından `32 dk` seviyesine çekildiğinde enerji kaybı ve yaşlanma hızla yükselirken, `w=[0.4, 0.2, 0.4]` bölgesinde sürenin ~%12–14 uzatılması enerji kaybını ~%12–14 ve yaşlanmayı ~%10 azaltmaktadır. Ayrıca düşük-orta akımlarda $E_{\text{loss}}$ ve $D$ güçlü pozitif korelasyon göstermiştir.
* **Açık Döngü Riskleri (Monte Carlo Bulgusu):** Deterministik SLSQP profili tam olarak $V_{\max} = 4.2000\text{ V}$ sınırına yaslandığı için, hücreden hücreye $R_0$ değişimi ($\sigma_{R_0} = 3.69\text{ m}\Omega$) altında açık döngü uygulandığında örneklerin %45'inde $4.202\text{ V}$ üstü hafif gerilim aşımı (`P95 = 4.224 V`) gözlenmiştir. Kapalı döngü CC-CV ise gerilim geri beslemesiyle %0 aşım sağlamıştır. Gerçek araç uygulamalarında optimal profilin bir gerilim emniyet payıyla (ör. $4.18\text{ V}$) veya kapalı döngü BMS geri beslemesiyle birlikte uygulanması gerekir.
* **Modelin Sınırlamaları:** (1) `OCV(SOC)` C/20 deşarj testinden türetildiği için şarj yönlü ~56 mV OCV histerezisi içerir. (2) $C_{\text{th}} = 134.86\text{ J/K}$ değeri test tutucularının ısıl ataletini de içerir; çekirdek–yüzey gradyanı ve entropik ısı ihmal edilmiştir. (3) $D(t)$ mutlak SOH tahmincisi değil, göreli elektro-termal stres ölçütüdür; lityum kaplama (lithium plating) anot potansiyeli açıkça modellenmemiştir.

---

# 8. Test Süreçleri

Tüm çekirdek modüller `pytest` altında **28 birim ve entegrasyon testi** ile doğrulanmıştır (tüm testler hatasız geçmiştir):

| Test Dosyası | Test Sayısı | Kapsam ve Beklenen Çıktı |
| :--- | :---: | :--- |
| `tests/test_data_loader.py` | 4 | Stanford/NASA gerçek dosya okuma, NaN/monotonluk tespiti, veri seti ayrımı |
| `tests/test_preprocessing.py` | 5 | `Step 0/0V` temizliği, NASA sıçrama/NaN temizliği, C/20 SOC doğrulama (`diff < 0.005`), `W10` sıcaklık kontrolü |
| `tests/test_battery_model.py` | 5 | Analitik RC (`< 1e-6 V`) ve termal soğuma (`< 1e-6 °C`), PCHIP OCV monotonluğu, çözücü uyumu |
| `tests/test_parameter_estimation.py` | 3 | Sentetik HPPC parametre geri kazanımı (`< %2`), fiziksel sınırlar, `W5/V4` RMSE `< 35 mV`, EIS uyumu |
| `tests/test_numerical_analysis.py` | 3 | Euler (`0.85–1.15`) ve RK4 (`3.7–4.3`) mertebesi, $\Delta t=130\text{ s}$ Euler kararsızlığı, merkezi fark `< 1e-9` |
| `tests/test_cc_cv.py` | 2 | $V_t \le 4.2005\text{ V}$ sınırlama, $|\text{SOC}(t_f)-0.90| < 10^{-5}$, `reference_scales.json` |
| `tests/test_optimizers.py` | 3 | GD/Adam/SLSQP kısıt uygunluğu, $J < 1.0$, sabit `seed=42` tekrarlanabilirliği |
| `tests/test_evaluation.py` | 3 | Sentetik Pareto cephesi, deterministik Monte Carlo örneklemesi (`seed=2026`), Aşama 10–11 çıktıları |

**Karşılaşılan Gerçek Sorunlar ve Giderilmesi:**
1. *OSF API Zaman Aşımı:* Standart OSF klasör gezgini yavaş/zaman aşımına uğradığı için WaterButler doğrudan uç noktasına geçildi ve veriler eksiksiz indirildi.
2. *İşaret Kuralı ve Eksik Sıcaklık:* `README.xlsx` şeması ile gerçek `.mat` akım işaretlerinin farklı olduğu sayısal incelemeyle saptanıp doğrulandı (`+` şarj); işlenmiş `.mat` dosyalarında sıcaklık bulunmadığı için ham `W10_Channel_6.2.xlsx` indirilip termal kalibrasyon yapıldı.
3. *Adam 2. Kademe Aşımı ve GD Seed Hassasiyeti:* Adam'da ceza geçişinde öğrenme oranının sıfırlanıp tekrar büyümesi `1.0125` sıçramasına yol açtığı için 2. kademe öğrenme oranı `0.25*alpha` ile ölçeklendi; GD'de `seed=42` takılmasını önlemek için `alpha0=2.0` ve `50` iterasyon uygulandı.
4. *Pareto Süre Darboğazı:* `i_min = 0.5 A` iken düşük-hız ağırlıklarında simülasyon adım sayısı büyüdüğü için Pareto taramasında `i_min = 1.0 A` ve `dt_opt_s = 20 s` (final doğrulama `1.0 s`) kullanılarak 20/20 çözüm ~75 s içinde elde edildi.

---

# 9. Sonuç ve Öneriler

Bu projede LG Chem INR21700-M50T hücresi için gerçek deney verilerinden kalibre edilmiş ve görülmemiş `W5/V4` hücreleri üzerinde `15.7–19.3 mV` RMSE ile bağımsız olarak doğrulanmış 1-RC elektro-termal-yaşlanma modeli geliştirilmiştir. Sayısal çözücü analizinde RK4 ($\mathcal{O}(h^4)$) yönteminin yüksek doğruluğu ve açık Euler kararsızlık sınırı ($120.30\text{ s}$) gösterilmiş; optimizasyon tarafında ise **SLSQP** algoritmasının hem hız (`1.97 s`) hem tekrarlanabilirlik (`std = 1.96e-7`) bakımından Gradient Descent ve Adam'a göre en verimli yöntem olduğu bulunmuştur. Dengeli senaryoda optimize edilen profiller, 1.0C CC-CV'ye kıyasla enerji kaybını `%12–14`, göreli yaşlanmayı `%9.5–10.8` azaltmıştır. Gelecek çalışmalarda: (i) şarj–deşarj OCV histerezisi ve SOC-bağımlı $R_0(\text{SOC})$ haritasının modele eklenmesi, (ii) Monte Carlo bulgusu doğrultusunda açık döngü gerilim aşımını sıfırlamak için şans kısıtlı (chance-constrained / robust) optimizasyon veya kapalı döngü MPC uygulanması ve (iii) elektrokimyasal anot potansiyeli kısıtıyla lityum kaplamanın doğrudan sınırlandırılması önerilmektedir.

---

# 10. Kaynakça

[1] G. Pozzato, A. Allam, and S. Onori, "Lithium-ion battery aging dataset based on electric vehicle real-driving profiles," *Data in Brief*, vol. 41, p. 107995, Apr. 2022, doi: 10.1016/j.dib.2022.107995.

[2] B. Saha and K. Goebel, "Battery Data Set," NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, USA, 2007. [Online]. Available: https://catalog.data.gov/dataset/li-ion-battery-aging-datasets

[3] LG Chem, "Product Specification: Rechargeable Lithium Ion Battery Model INR21700 M50T 4.85Ah," LG Chem, Ltd., Technical Datasheet.

[4] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, San Diego, CA, USA, 2015, arXiv:1412.6980.

[5] D. Kraft, "A software package for sequential quadratic programming," German Aerospace Research Establishment (DFVLR), Cologne, Germany, Tech. Rep. DFVLR-FB 88-28, 1988.

[6] P. Virtanen et al., "SciPy 1.0: Fundamental algorithms for scientific computing in Python," *Nature Methods*, vol. 17, no. 3, pp. 261–272, Mar. 2020, doi: 10.1038/s41592-019-0686-2.

[7] G. L. Plett, *Battery Management Systems, Volume I: Battery Modeling*. Norwood, MA, USA: Artech House, 2015.
