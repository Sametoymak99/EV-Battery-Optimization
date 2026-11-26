# Proje Önerisi: Elektrikli Araç Bataryalarında Çok Amaçlı Optimal Şarj Stratejisinin Nümerik Optimizasyonu

**Ders:** MAT353 – Nümerik Analiz (2026–2027 Güz Dönemi) | **Öğretim Üyesi:** Doç. Dr. Murat GEZER | **Öğrenci:** Samet Oymak (24120205042)

## 1. Problem Tanımı ve Amaç
Elektrikli araçlarda lityum iyon bataryaların hızlı şarj edilmesi kullanıcı deneyimi açısından kritik olsa da yüksek şarj akımları iç direnç kaynaklı enerji kayıplarını, ısınmayı ve batarya yaşlanmasını hızlandırır. Bu projede, **LG Chem INR21700-M50T** hücre verileri temel alınarak batarya dinamiği birinci dereceden Thevenin eşdeğer devresi, birleştirilmiş (lumped) termal model ve ampirik yaşlanma göstergesi ile modellenecek; şarj süresi ($J_1$), enerji kaybı ($J_2$) ve göreli yaşlanma göstergesini ($J_3$) birlikte minimize eden optimal şarj akımı profili $I(t)$ nümerik optimizasyon yöntemleriyle elde edilecektir.

## 2. Veri Kaynakları
* **Ana Veri Seti:** Stanford Energy Control Laboratory INR21700-M50T Veri Seti (`https://osf.io/qsabn/`).
* **Yardımcı Veri Seti:** NASA Ames PCoE Li-ion Battery Aging Veri Seti (`catalog.data.gov`).
* İki veri seti farklı hücre kimyası, form faktörü (21700 vs. 18650) ve test protokolüne sahip olduğu için doğrudan birleştirilmeyecek; NASA verileri yalnızca bağımsız hücrede gerilim/sıcaklık/yaşlanma davranışının nitel kontrolünde kullanılacaktır.

## 3. Yöntem ve Nümerik Kapsam
1. **Diferansiyel Denklem Çözümü:** Euler, 4. Derece Runge-Kutta (RK4) ve SciPy `solve_ivp` yöntemlerinin doğruluk, adım boyu duyarlılığı ve çalışma süresi karşılaştırması.
2. **Parametre Tahmini:** Doğrusal olmayan en küçük kareler ve interpolasyon teknikleri ile $Q$, $\text{OCV}(\text{SOC})$, $R_0$, $R_1$, $C_1$, $R_{\text{th}}$, $C_{\text{th}}$ kestirimi; eğitim ve doğrulama setlerinin veri sızıntısı olmadan ayrılması.
3. **Nümerik Optimizasyon:** Sonlu fark gradyanlı Gradient Descent, Adam Optimizer (kısıtlar için projeksiyon ve ceza fonksiyonu) ve SciPy SLSQP algoritmalarının CC-CV referans stratejisiyle kıyaslanması; Big-O karmaşıklık ve çalışma zamanı analizi.
4. **Pareto ve Monte Carlo Analizi:** Çok amaçlı ağırlık taraması ile Pareto cephesinin çıkarılması; parametre belirsizlikleri altında sabit seed ile tekrarlanabilir Monte Carlo simülasyonları (ortalama ± standart sapma).

## 4. 12 Aşamalı Geliştirme Takvimi
1. Veri setlerini inceleme ve şema analizi *(Mevcut Aşama)*
2. Veri temizleme, şarj döngülerini ayırma ve görselleştirme
3. Matematiksel modelin kurulması
4. Deneysel verilerden parametre tahmini ve bağımsız doğrulama
5. Euler, RK4 ve `solve_ivp` karşılaştırması
6. CC-CV referans şarj simülasyonu
7. Gradient Descent uygulaması
8. Adam uygulaması
9. SLSQP uygulaması
10. Çok amaçlı optimizasyon ve Pareto analizi
11. Monte Carlo belirsizlik analizi
12. Sonuçların karşılaştırılması, 8–15 sayfalık MAT353 raporu ve 10 slaytlık sunum hazırlanması
