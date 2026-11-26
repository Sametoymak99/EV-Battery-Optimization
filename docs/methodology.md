# Metodoloji ve Sayısal Analiz Yaklaşımı

## 1. Veri Yönetimi ve Veri Sızıntısının Önlenmesi
* **Veri Ayrımı:** Ana modelleme için Stanford Energy Control Laboratory'nin LG Chem INR21700-M50T veri seti [1] kullanılmıştır. Kalibrasyon ve doğrulama hücreleri kesin biçimde ayrılmıştır:
  * **Eğitim / Kalibrasyon Seti:** `W4` (C/4), `W8` (C/2), `W9` (1C) ve `W10` (3C) hücrelerinin `Diag_1` (C/20 kapasite ve HPPC) testleri ile `W10_Channel_6.2.xlsx` ham 3C termal verisi.
  * **Bağımsız Doğrulama Seti:** Kalibrasyona hiç dahil edilmeyen `W5` (C/2, `Diag_2` HPPC) ve `V4` (C/4, `Diag_1` HPPC) hücreleri ile `Cycling_1` (Çevrim 2) şarj profilleri.
  * **Yardımcı Veri Seti (Birleştirilmez):** NASA Ames PCoE 18650 veri seti (`B0005`, `B0006`, `B0007`, `B0018`, `B0029`, `B0053`) [2], farklı hücre kimyası ve form faktörü nedeniyle Stanford verisiyle birleştirilmemiş; yalnızca gerilim/sıcaklık eğilimi ve empedans artışı kontrolünde kullanılmıştır.
* **Veri Temizliği:** Stanford çevrim dosyalarındaki `Step 0 / 0 V` artifaktları, zaman monotonluğu bozulmaları ve NASA dosyalarındaki `NaN` ile tekil gerilim sıçramaları (`0.003–8.39 V`) filtrelenmiştir.

## 2. Parametre Kestirimi Metodolojisi
* **OCV(SOC) ve Kapasite ($Q$):** Kalibrasyon hücrelerinin `Diag_1` C/20 eğrilerinden yamuk (trapezoidal) Coulomb sayımıyla doğrulanan SOC değerleri üzerine monotonik şekil-korumalı PCHIP interpolasyonu uygulanmıştır.
* **1-RC Elektriksel Parametreler ($R_0, R_1, C_1$):** `HPPC_test` verisinin `0.15 <= SOC <= 0.95` penceresinde parça-sabit analitik RC güncellemesi ve `scipy.optimize.least_squares` (`method='trf'`) kullanılarak kestirilmiş; `EIS_test` ($\text{Im}(Z) \approx 0$) ile $R_0$ çapraz doğrulanmıştır.
* **Birleştirilmiş Termal Parametreler ($R_{\text{th}}, C_{\text{th}}$):** `W10_Channel_6.2` 3C ısınma + dinlenme eğrisi üzerinde Joule ısı üretimi $P_{\text{loss}}(t) = I^2 R_0 + V_p^2 / R_1$ kullanılarak doğrusal olmayan en küçük kareler uygulanmıştır.
* **Yaşlanma Modeli Sınırı:** Arrhenius ve C-oranı üssü parametreleri $(k_0, E_a, p, \alpha_{\text{soc}})$ literatür temelli göreli indeks olarak tanımlanmıştır; $D(t)$ mutlak SOH yerine göreli elektro-termal stres ölçütü olarak raporlanır.

## 3. Sayısal Çözüm ve Optimizasyon Yöntemleri
* **ODE Entegrasyonu:** İleri Euler ($\mathcal{O}(h)$), klasik 4. derece Runge–Kutta (RK4, $\mathcal{O}(h^4)$) ve SciPy `solve_ivp` (RK45), yüksek hassasiyetli `DOP853` (`rtol=1e-12`) referansına karşı kıyaslanmıştır. Açık Euler kararlılık sınırı $2\min(R_1 C_1, R_{\text{th}} C_{\text{th}}) = 120.30\text{ s}$ olarak belirlenmiştir.
* **Kontrol Parametrizasyonu:** Serbest bitiş zamanı ($t_f$) problemini türevlenebilir kılmak ve $\text{SOC}(t_f) = 0.90$ hedefini otomatik sağlamak için $[0.20, 0.90]$ aralığı $N=10$ eşit $\Delta\text{SOC} = 0.07$ dilimine bölünmüş ve karar vektörü $\mathbf{u} = [I_1, \dots, I_{10}]^T \in [0.50, 9.72]\text{ A}$ seçilmiştir.
* **Optimizasyon Algoritmaları:**
  * **Gradient Descent:** İleri sonlu fark gradyanı ($h = 10^{-4}\text{ A}$), Armijo geri izlemeli adım araması, kutu projeksiyonu ve 2-kademeli karesel dış ceza ($\rho_V: 500 \to 5000$).
  * **Adam Optimizer:** Birinci/ikinci moment kestirimleri ($\beta_1=0.9, \beta_2=0.999, \varepsilon=10^{-8}$), kosinüs öğrenme oranı azalımı, kutu projeksiyonu ve 2-kademeli dış ceza.
  * **SLSQP:** Kutu sınırları ve aşama bazlı gerilim/sıcaklık eşitsizlik kısıtlarıyla doğrudan sıralı karesel programlama [5], [6].
* **Big-O Karmaşıklık:** $M$ ODE zaman adımı ve $N=10$ karar değişkeni için tek simülasyon $\mathcal{O}(M)$, sonlu fark gradyanı $\mathcal{O}(N \cdot M)$ ve $K$ iterasyonlu optimizasyon $\mathcal{O}(K \cdot N \cdot M)$ maliyetindedir.

## 4. Pareto ve Monte Carlo Analizi
* **Pareto Analizi:** Farklı ağırlık üçlüleri $(w_1, w_2, w_3)$ ($\sum w_i = 1$) taranarak baskın olmayan (non-dominated) çözümler çıkarılır ve geleneksel CC-CV ($0.5\text{C}\text{–}2.0\text{C}$) eğrisiyle kıyaslanır.
* **Monte Carlo Belirsizlik Analizi:** Sabit tohum değeri (`seed=2026`) ile $N_{\text{MC}}=200$ örneklem üzerinde $(R_0, R_1, C_1, Q, R_{\text{th}}, C_{\text{th}})$ parametre belirsizliklerinin amaçlara ve kısıt aşım olasılıklarına etkisi **ortalama ± standart sapma** biçiminde hesaplanır.
