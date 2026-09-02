🏛️ Kurumsal Finansal Veri Bilimi & Aktüeryal Laboratuvarı
Sigortacılık, risk yönetimi, varlık-yükümlülük yönetimi (ALM) ve finansal skorlama alanlarındaki karmaşık matematiksel modelleri somutlaştırmak, gerçek veri setleriyle test etmek ve endüstriyel standartlarda raporlamak amacıyla geliştirilmiş uçtan uca bir kurumsal karar destek sistemidir.

🚀 Öne Çıkan Özellikler ve Mimari Yapı
Dinamik Veri Yükleme Motoru (st.file_uploader): Sadece hazır açık kaynak verilerle sınırlı kalmayıp; kullanıcıların kendi CSV veya Excel dosyalarını yükleyerek modelleri anlık olarak yeniden eğitmesine ve test etmesine olanak tanır.
Aktüeryal Fiyatlama & Risk Modülleri: Genelleştirilmiş Doğrusal Modeller (GLM) ve regresyon yaklaşımlarıyla saf prim (Pure Premium) tahmini ve hasar frekans analizleri.
Varlık-Yükümlülük Yönetimi (ALM) & Raporlama: Nakit akışı eşitleme simülasyonları ve XlsxWriter motoruyla anlık kurumsal Excel raporu dışa aktarımı.
İlişkisel Veritabanı Loglama: Gerçekleşen tüm simülasyonların ve fiyatlama kararlarının SQLite veritabanında kalıcı olarak saklanması ve geçmiş takibi.
Kurumsal Tasarım Dili: Saf beyaz üst bar, koyu lacivert yan menü, minimal kurumsal simgeler ve teorik altyapı expander pencereleriyle optimize edilmiş UI/UX.
📊 Kaggle Veri Setleri ile Entegrasyon ve Test Senaryoları
Bu platformu kendi verilerinle veya dünyaca ünlü açık kaynaklı Kaggle veri setleriyle test edebilirsin. Önerilen test senaryoları:

Kasko Saf Prim Fiyatlama Modülü için:
Kaggle Veri Seti: French Motor Third-Party Liability Claims (freMTPL2freq)
Nasıl Test Edilir?: İlgili CSV dosyasını kasko modülündeki dosya yükleme alanına bırakarak modelin sürücü yaşı ve motor gücüne göre saf prim hesaplamasını gözlemleyin.
Otomatik Kredi Risk Skorlama Modülü için:
Kaggle Veri Seti: Home Credit Default Risk veya German Credit Data
Nasıl Test Edilir?: Müşteri finansal verilerini yükleyerek bankacılık temerrüt olasılık skorlamasını çalıştırın.
Müşteri Kaybı (Churn) Erken Uyarı Modülü için:
Kaggle Veri Seti: Bank Customer Churn Prediction
Nasıl Test Edilir?: Banka müşteri terk davranışlarını analiz edin.

🛠️ Kurulum ve Çalıştırma
Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin:

# Repoyu klonlayın
git clone https://github.com/kullanici-adin/finansal-aktueryal-lab.git
cd finansal-aktueryal-lab

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# Uygulamayı başlatın
streamlit run app.py

👩‍💻 Geliştirici
Sultan Kuş | İstanbul Üniversitesi Matematik | Finansal Veri Bilimi & Aktüerya
