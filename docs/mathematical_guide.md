# Matematiksel Model ve Nümerik Çözüm Kılavuzu

Bu doküman, LG Chem **INR21700-M50T** lityum iyon bataryanın şarj sürecini modellemek ve çok amaçlı optimal şarj akımı profilini belirlemek için kullanılan fiziksel denklemleri, kısıtları, amaç fonksiyonlarını ve sayısal entegrasyon/karmaşıklık temellerini açıklar.

---

## 1. Durum Uzayı ve Kontrol Değişkeni

* **Durum vektörü:**
  $$x(t) = \begin{bmatrix} \text{SOC}(t) \\ V_p(t) \\ T(t) \\ D(t) \end{bmatrix}$$
  * $\text{SOC}(t) \in [0, 1]$: Batarya doluluk oranı (birimsiz).
  * $V_p(t)$ [V]: 1-RC polarizasyon kolu üzerindeki dinamik aşırı gerilim.
  * $T(t)$ [°C]: Birleştirilmiş (lumped) hücre sıcaklığı.
  * $D(t)$ [-]: Göreli yaşlanma göstergesi (birimsiz).
* **Kontrol değişkeni:** $u(t) = I(t)$ [A], şarj yönünde pozitif ($I(t) \ge 0$).
* **Ölçüm / Çıkış değişkeni:** Terminal gerilimi $V_t(t)$ [V].

---

## 2. Sistem Denklemleri

### 2.1. Doluluk Oranı (SOC) Dinamiği
$$\frac{d\text{SOC}}{dt} = \frac{\eta\, I(t)}{3600\, Q}$$
* $\eta \in (0, 1]$: Coulombik şarj verimliliği (birimsiz).
* $Q$ [Ah]: Hücre kapasitesi.
* $3600$ [s/h]: Saat–saniye birim dönüşüm katsayısı.
* **Optimizasyondaki rolü:** Şarj sürecinin $\text{SOC}(0) = 0.20$ başlangıç durumundan $\text{SOC}(t_f) = 0.90$ hedef durumuna ulaşmasını izler.

### 2.2. Birinci Dereceden Thevenin Polarizasyon Gerilimi
$$\frac{dV_p}{dt} = -\frac{V_p(t)}{R_1 C_1} + \frac{I(t)}{C_1}$$
* $R_1$ [$\Omega$]: Polarizasyon (yük transferi / difüzyon) direnci.
* $C_1$ [F]: Polarizasyon kapasitesi ($\tau_1 = R_1 C_1$ [s] gevşeme zaman sabiti).
* **Optimizasyondaki rolü:** Bataryanın akım değişimlerine anlık yerine üstel bir zaman sabiti $\tau_1$ ile tepki vermesini yakalar; hem terminal gerilimi artışını hem polarizasyon güç kaybını belirler.

### 2.3. Terminal Gerilimi
$$V_t(t) = \text{OCV}(\text{SOC}(t)) + I(t)\, R_0 + V_p(t)$$
* $\text{OCV}(\text{SOC})$ [V]: C/20 tanı testinden elde edilen şekil-korumalı PCHIP açık devre gerilimi eğrisi.
* $R_0$ [$\Omega$]: Anlık ohmik iç direnç.
* **Optimizasyondaki rolü:** Şarj sırasında $V_t(t) \le V_{\max}$ ($4.20\text{ V}$) güvenlik ve lityum kaplama önleme sınırının denetlenmesini sağlar.

### 2.4. Birleştirilmiş (Lumped) Termal Model
$$C_{\text{th}} \frac{dT}{dt} = I(t)^2 R_0 + \frac{V_p(t)^2}{R_1} - \frac{T(t) - T_{\text{amb}}}{R_{\text{th}}}$$
* $C_{\text{th}}$ [J/K]: Hücre ısıl sığası.
* $R_{\text{th}}$ [K/W]: Hücre yüzeyi ile ortam arasındaki ısıl direnç ($\tau_{\text{th}} = R_{\text{th}} C_{\text{th}}$ [s]).
* $T_{\text{amb}}$ [°C]: Ortam sıcaklığı (Stanford kabini için $23.0\ ^\circ\text{C}$).
* $I^2 R_0 + V_p^2 / R_1$ [W]: İç direnç ve polarizasyon kolunda oluşan toplam Joule ısınma gücü.
* **Varsayım ve sınırlamalar:** Entropik tersinir ısı ($I T \frac{\partial U}{\partial T}$), hücre içi radyal çekirdek–yüzey sıcaklık gradyanı ve komşu hücreler arası ısıl etkileşim başlangıç modelinde ihmal edilmiştir.

### 2.5. Ampirik Göreli Yaşlanma Modeli
$$\frac{dD}{dt} = k_0 \exp\!\left(-\frac{E_a}{R_g\, T_K(t)}\right) \left(\frac{|I(t)|}{Q}\right)^p g(\text{SOC}(t)), \qquad T_K(t) = T(t) + 273.15$$
$$g(\text{SOC}) = 1 + \alpha_{\text{soc}}\, \text{SOC}^2$$
* $k_0$: Deneysel/literatür ölçek katsayısı, $E_a$ [J/mol]: Aktivasyon enerjisi, $R_g = 8.3144626\text{ J/(mol}\cdot\text{K)}$: Evrensel gaz sabiti, $p$: C-oranı üssü.
* **Önemli not:** $D(t)$ tek başına mutlak kapasite kaybını (SOH) kanıtlamaz; yalnızca farklı şarj profillerinin göreceli elektro-termal stresini karşılaştırmak için **göreli yaşlanma göstergesi** olarak kullanılır.

### 2.6. Direnç Kaynaklı Enerji Kaybı
$$E_{\text{loss}} = \int_0^{t_f} \left(I(t)^2 R_0 + \frac{V_p(t)^2}{R_1}\right) dt \quad [\text{J}]$$

---

## 3. Çok Amaçlı Optimizasyon Formülasyonu ve Kontrol Parametrizasyonu

### 3.1. Normalize Edilmiş Amaç Fonksiyonu
$$J_1 = \frac{t_f}{t_{\text{ref}}}, \qquad J_2 = \frac{E_{\text{loss}}}{E_{\text{ref}}}, \qquad J_3 = \frac{D(t_f) - D(0)}{D_{\text{ref}}}$$
$$\min_{\mathbf{u}}\; J(\mathbf{u}) = w_1 J_1 + w_2 J_2 + w_3 J_3, \qquad w_1 + w_2 + w_3 = 1,\quad w_i \ge 0$$
Referans değerler ($t_{\text{ref}}, E_{\text{ref}}, D_{\text{ref}}$) Aşama 6'da kurulacak geleneksel **CC-CV referans stratejisinden** hesaplanacaktır.

### 3.2. Kısıtlar (Simülasyon Boyunca)
* $\text{SOC}(0) = 0.20$, $\quad \text{SOC}(t_f) = 0.90$, $\quad 0 \le \text{SOC}(t) \le 1$
* $0 < I_{\min} \le I(t) \le I_{\max}$
* $V_t(t) \le V_{\max} = 4.20\text{ V}$
* $T(t) \le T_{\max}$ (ör. $45.0\ ^\circ\text{C}$)

### 3.3. SOC-Tabanlı Çok Aşamalı Akım Parametrizasyonu (Serbest $t_f$ Çözümü)
Sabit zaman adımlı ayrıklaştırmada $\text{SOC}(t) = 0.90$'a ulaşma anı $t_f$ kesikli ve türevlenemeyen bir yapı oluşturur. Bunu önlemek için hedef doluluk aralığı $[\text{SOC}_0, \text{SOC}_f] = [0.20, 0.90]$ eşit $N$ adet $\Delta \text{SOC}_k$ dilimine bölünür ve karar vektörü $\mathbf{u} = [I_1, I_2, \dots, I_N]^T$ seçilir:
$$\Delta t_k(I_k) = \frac{3600\, Q\, \Delta\text{SOC}_k}{\eta\, I_k}, \qquad t_f(\mathbf{u}) = \sum_{k=1}^{N} \Delta t_k(I_k)$$
Bu formülasyonla $t_f(\mathbf{u})$ karar değişkenlerine göre $C^\infty$ türevlenebilir olur ve $\text{SOC}(t_f) = 0.90$ hedef kısıtı **otomatik ve tam olarak** sağlanır.

---

## 4. Sayısal Çözüm Yöntemleri, Kararlılık ve Karmaşıklık

* **İleri Euler Yöntemi ($O(h)$ global hata):** $y_{n+1} = y_n + h f(t_n, y_n)$. Doğrusal kararlılık sınırı:
  $$h < 2 \min\!\left(R_1 C_1,\; R_{\text{th}} C_{\text{th}}\right)$$
* **4. Derece Runge–Kutta (RK4, $O(h^4)$ global hata):** Her adımda 4 fonksiyon değerlendirmesi ($k_1, k_2, k_3, k_4$) ile yüksek doğruluk sağlar.
* **SciPy `solve_ivp` (RK45 / DOP853):** Uyarlanabilir adım boyutlu referans çözücü.
* **Algoritmik Karmaşıklık (Big-O):** $M$ toplam zaman adımı ve $N$ aşama akımı değişkeni için:
  * Tek bir simülasyon: $\mathcal{O}(M)$
  * Sonlu fark gradyan değerlendirmesi: $\mathcal{O}(N \cdot M)$
  * $K$ iterasyonlu GD / Adam optimizasyonu: $\mathcal{O}(K \cdot N \cdot M)$
