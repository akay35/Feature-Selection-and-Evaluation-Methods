# Feature-Selection-and-Evaluation-Methods
## Feature Selection and Evaluation Methods (Özellik Seçimi ve Değerlendirme Yöntemleri)

Veri ön işleme ve özellik seçimi, makine öğrenmesi modellerinin başarısının temel taşlarıdır. Harvard Business Review'da 2018 yılında yer alan “Eğer veriniz kötü ise makine öğrenmesi araçlarınız kullanışsız olacaktır” ifadesi, model performansının büyük ölçüde verinin kalitesine bağlı olduğunu net bir şekilde ortaya koymaktadır. Bu durum, özellikle büyük veri setlerinde özellik seçiminin ve doğru değişken mühendisliğinin gerekliliğini daha da artırmaktadır.

Makine öğrenmesi modellerinde kullanılan tüm özellikler, tahmin gücüne aynı derecede katkıda bulunmaz. Gereksiz veya fazla korrele olmuş özellikler, hem modelin performansını düşürebilir hem de aşırı uyum (overfitting) problemlerine yol açabilir. Variance Inflation Factor (VIF), bu tür durumlarda devreye girerek, çoklu doğrusal bağlantı sorunlarını tespit etmemize ve gereksiz özellikleri ortadan kaldırmamıza yardımcı olur. Bu süreç, veriyi daha yalın hale getirirken, modelin güvenilirliğini de artırır. 

Bunun yanı sıra, makine öğrenmesi modellerinin başarısı yalnızca veri kalitesiyle değil, aynı zamanda veriden anlamlı değişkenler türetme becerisiyle de ilişkilidir. Bu noktada, Recursive Feature Elimination (RFE) gibi teknikler, en etkili özellikleri seçerek modelin karmaşıklığını azaltır ve performansını optimize eder. Yapay zeka alanının öncülerinden Andrew Ng'in “Applied machine learning is basically feature engineering” sözü, değişken mühendisliğinin bu süreçteki önemini açıkça ortaya koymaktadır.

Feature Importance gibi metrikler, modelin hangi değişkenlerden en fazla faydalandığını belirleyerek, hem tahmin gücünü artırır hem de modelin karar mekanizmalarını daha anlaşılır hale getirir. 

Bu metriklerin ve yöntemlerin genel adı, özellik seçimi ve değerlendirme yöntemleri olarak tabir edilebilir. Amaçları, model performansını artırmak, daha açıklayıcı ve anlamlı modeller oluşturmak ve gereksiz veya problemli özelliklerin etkisini azaltmaktır.

---

### VIF (Variance Inflation Factor) Nedir?
VIF, değişkenler arasındaki çoklu doğrusal bağlantıyı (multicollinearity) derecesini ölçmek için kullanılır. 
VIF değeri yüksek olan bir değişken, diğer değişkenlerle yüksek oranda ilişkili demektir.
Yani VIF, bir bağımsız değişkenin diğer bağımsız değişkenlerle ne kadar ilişkili olduğunu değerlendirir. 
VIF değeri yüksekse, o değişken diğer değişkenlerle güçlü bir şekilde ilişkilidir, bu da model performansını olumsuz etkileyebilir.

#### VIF değerleri şu şekilde yorumlanabilir:

- VIF < 5: Çoklu doğrusal bağlantı sorunu yoktur.
- 5 ≤ VIF < 10: Çoklu doğrusal bağlantı kısmen yüksektir. Dikkat edilmelidir.
- VIF ≥ 10: Çoklu doğrusal bağlantı yüksektir ve bu değişkeni modelden çıkarmayı veya dönüşüm uygulamayı düşünmelisiniz.

![image](https://github.com/akay35/Feature-Selection-and-Evaluation-Methods/blob/main/VIF%20Scores.png)

#### VIF Ne Zaman Hesaplanmalıdır?
VIF analizi, model öncesi ve model sonrası yapılabilir, ancak genellikle model öncesinde yapılması daha yaygındır.

##### 1. Model Öncesi Analiz
Amaç: Bağımsız değişkenler arasında çoklu doğrusal bağlantıyı tespit etmek ve modeli optimize etmektir.
Ne Zaman Yapılır?: Özellikle regresyon modelleri oluşturulmadan önce, 
değişkenlerin bağımsız olduğundan emin olmak için VIF analizi yapılır. 
Yüksek VIF değerine sahip değişkenler modelden çıkarılarak overfitting ve yanıltıcı sonuçların önüne geçilebilir.
Örnek Durum: Özellikle lineer regresyon gibi doğrusal modeller çoklu doğrusal bağlantıya karşı hassastır. 
Bu nedenle, bağımsız değişkenlerin birbirleriyle ilişkisinin düşük olduğundan emin olmak için VIF hesaplanmalıdır.

##### 2. Model Sonrası Analiz
Amaç: Modelin performansını değerlendirdikten sonra, çoklu doğrusal bağlantı olup olmadığını kontrol etmek ve gerekirse değişiklikler yapmaktır.
Ne Zaman Yapılır?: Model eğitildikten sonra hala düşük performans veya yüksek hata oranları (örneğin, yüksek RMSE veya düşük R²) gözlemleniyorsa, bu durumda VIF analizi yapılabilir.
Örnek Durum: Özellikle düşük performans sergileyen veya aşırı uyum gösteren modellerde,  VIF kullanılarak hangi değişkenlerin modeli etkileyebileceği tespit edilir ve gerekirse model optimize edilir.

#### VIF Analizinden Sonra Neler Yapılabilir?
VIF değeri yüksek olan değişkenler tespit edildiyse, şu çözümler uygulanabilir:

- Yüksek VIF değerine sahip değişkenleri çıkarılması: Çoklu doğrusal bağlantıya neden olan değişkenler modelden çıkartılabilir.
- Dönüşümler uygulanası: Örneğin, değişkenler arasında logaritmik dönüşümler veya standartlaştırma yaparak bağlantı azaltılabilir.
- PCA (Principal Component Analysis): Yüksek korelasyona sahip değişkenleri, ana bileşen analizi ile daha az sayıda bağımsız değişkene indirgemek etkili olabilir.
- Feature Engineering: Değişkenleri yeniden tanımlayarak veya birleştirerek yeni değişkenler oluşturulabilir.

###### Statsmodels kütüphanesinin outliers_influence modülünden variance_inflation_factor fonksiyonu import edilerek VIF kullanılabilir. 
###### from statsmodels.stats.outliers_influence import variance_inflation_factor

---

### RFE (Recursive Feature Elimination) Nedir?
Recursive Feature Elimination (RFE), özellik seçimi (feature selection) için kullanılan bir tekniktir. 
Özellikle gereksiz veya düşük önem düzeyine sahip özellikleri belirleyerek modelin performansını artırmayı amaçlar. 
RFE, modelin başarısına en fazla katkıda bulunan özellikleri bulmak için tekrarlayan bir şekilde özellikleri eler.

#### RFE Nasıl Çalışır?
- Model Eğitimi: Başlangıçta, tüm özelliklerle bir model eğitilir (genellikle regresyon veya sınıflandırma modeli).
- Özellik Önem Sıralaması: Model, her bir özelliğin önem derecesini hesaplar.
- Özellik Elemek: En düşük önem düzeyine sahip olan bir veya birden fazla özellik çıkarılır.
- Tekrar: Kalan özelliklerle model yeniden eğitilir ve önem dereceleri tekrar hesaplanır.
- Durdurma Kriteri: Bu işlem, önceden belirlenmiş bir sayıda özellik kalana kadar veya model performansı belirli bir seviyeye ulaşana kadar devam eder.
- Sonuç olarak, RFE özelliklerin önem sırasını belirler ve modelin en iyi performans gösterdiği özellik setini sunar.

![image](https://github.com/akay35/Feature-Selection-and-Evaluation-Methods/blob/main/RFE.png)

RFE, belirtilen sayıda en iyi özellikleri seçmek için iteratif bir süreç uygular. Her iterasyonda, model performansını en az etkileyen özelliği kaldırır.
En önemli özelliklere 1 derecesi atanır, çünkü bunlar modelin tahmin performansı için kritik kabul edilen özelliklerdir.
Daha az önemli özelliklere ise sırasıyla 2, 3, 4 vb. dereceler atanır. Bu sıralama, bu özelliklerin ne kadar az etkili olduğunu gösterir.

#### RFE Model Öncesi mi Yoksa Model Sonrası mı Hesaplanır?
RFE, model oluşturulmadan önce, modelin performansını artırmak amacıyla uygulanır. Yani, model öncesi bir adımdır.

Model öncesi adım olarak, modelin başarısız olmasına neden olabilecek gereksiz veya düşük bilgi içeriğine sahip özellikleri ortadan kaldırarak modelin daha iyi genelleme yapmasını sağlar.
Model sonrası yapılması anlamlı değildir, çünkü model zaten tüm özelliklerle eğitilmiş olacaktır. 
Dolayısıyla, modelin eğitilmesi sırasında düşük performansa neden olabilecek özellikler zaten dahil edilmiştir.

#### RFE'nin Önemi ve Sağladığı Faydalar
Özellik Seçimi ve Model Basitleştirme:

Gereksiz özellikleri çıkararak modeli daha basit hale getirir.
Daha az özellik kullanarak modeli eğitmek, eğitim süresini kısaltır ve hesaplama maliyetlerini azaltır.
Çok sayıda gereksiz veya gürültülü değişken, modelin overfitting yapmasına neden olabilir. 
RFE, yalnızca en önemli değişkenleri seçerek overfitting riskini azaltır.
Model, yalnızca en önemli değişkenleri kullanarak eğitildiğinde, genellikle daha iyi genelleme performansı gösterir.
Daha az ancak daha anlamlı özellik kullanıldığında, modelin çıktıları daha kolay yorumlanabilir hale gelir.

###### Scikit learn kütüphanesi içerisinde özellik seçimi için kullanılan feature_selection modulünden RFE import edilebilir.
###### from sklearn.feature_selection import RFE

---

### Feature Importance Nedir?
Feature Importance (Özellik Önem Derecesi), bir makine öğrenimi modelinde bağımsız değişkenlerin hedef değişken üzerindeki etkisini ölçen bir metriktir. Özellikle, modelin tahmin performansını en çok etkileyen değişkenleri belirlemek için kullanılır.

#### Feature Importance Ne Zaman Hesaplanmalıdır?
Feature Importance analizi hem model öncesi hem de model sonrası yapılabilir.

##### 1. Model Öncesi Analiz
Modelin tahmin gücüne en çok katkıda bulunan değişkenleri belirlemek ve önemsiz değişkenleri elemek. Gereksiz özellikleri çıkarmak, modelin daha hızlı çalışmasını sağlar ve aşırı öğrenme (overfitting) riskini azaltır.

Model oluşturulmadan önce, fazla özellik sayısını azaltmak veya en anlamlı değişkenleri seçmek istenildiğinde uygulanabilir.

##### 2. Model Sonrası Analiz
Model eğitildikten sonra hangi özelliklerin modele katkı sağladığını değerlendirmek. Bu analiz, model optimizasyonu veya modelin yorumlanabilirliğini artırmak için kullanılabilir.

Model eğitildikten sonra, tahmin performansını etkileyen özellikleri analiz etmek veya belirli özelliklerin etkisini anlamak istenildiğinde uygulanabilir..

![image](https://github.com/akay35/Feature-Selection-and-Evaluation-Methods/blob/main/Feature%20Importance.png)

Grafiği, modelin tahmin performansında en önemli değişkenleri ve önem derecelerini göstermektedir.
Grafik, CLTV'nin açık ara modelin tahmin performansını etkileyen en önemli değişken olduğunu gösteriyor. 
İkinci ve üçüncü sırada yer alan bu değişkenler, müşterilerin ödeme düzenliliği ve müşteri sadakat süresi ile ilgili önemli içgörüler sunuyor.
Alt sıralardaki değişkenler  model için daha az kritik önem arz etmektedir. Bunlar modelden çıkarılabilir veya yeniden düzenlenebilir.

### ÖRNEK ÇALIŞMA

Bu çalışmada kullanılan veri setinde; Telco müşteri kaybı verileri, Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan bir telekom şirketi hakkında bilgiler içermektedir..

Bu çalışmada, veri setinde bulunan özelliklerin model performansına olan etkisini artırmak ve daha sade bir model oluşturmak amacıyla değişken seçimi ve önem sıralaması üzerinde durulmuştur. İlk olarak, veri ön işleme adımları gerçekleştirilmiş; eksik değerler doldurulmuş, kategorik değişkenler kodlanmış ve ölçeklendirme işlemleri yapılmıştır. Daha sonra, çoklu doğrusal ilişki (multicollinearity) problemini tespit etmek ve çözmek amacıyla VIF (Variance Inflation Factor) analizi uygulanmıştır. VIF değerleri yüksek olan değişkenler belirlenmiş ve bu değişkenler modelden çıkarılmıştır.

![image](https://github.com/akay35/Feature-Selection-and-Evaluation-Methods/blob/main/vifcode.png)

Devamında, değişkenlerin model için önem sırasını belirlemek amacıyla RFE (Recursive Feature Elimination) yöntemi kullanılmıştır. Bu yöntem, modelin en iyi performansı gösterebilmesi için gereksiz veya düşük öneme sahip özellikleri elemeyi hedeflemiştir. 

![image](https://github.com/akay35/Feature-Selection-and-Evaluation-Methods/blob/main/rfecode.png)

LightGBM ile model eğitildikten sonra, seçilen değişkenlerin önem derecelerini daha iyi anlamak için feature importance analizine başvurulmuştur. Bu adımda, modelin her bir değişkene atfettiği ağırlıklar incelenerek en etkili özellikler sıralanmıştır.

Son olarak, bu süreçte elde edilen sonuçlar değerlendirilmiş, seçilen değişkenler kullanılarak model yeniden oluşturulmuş ve performans metrikleri ile değerlendirilmiştir. Bu sayede, hem modelin performansı artırılmış hem de daha açıklanabilir ve sade bir yapı elde edilmiştir. Bu yaklaşım, değişken seçim sürecinde kullanılan yöntemlerin birbirini desteklemesiyle daha güvenilir ve sağlam bir sonuç sağlamıştır.

### Değerlendirme

VIF, RFE ve Feature Importance gibi metrikleri kullanmak, modelin gereksiz ve fazla karmaşık özelliklerden arınmasını sağlayabilir, böylece sadece gerçekten değerli bilgiyi kullanarak daha verimli ve etkili tahminler yapılmasına olanak tanır. Bu süreç, modelin doğruluğunu artırırken, aynı zamanda yorumlanabilirliği artırır, çünkü hangi değişkenlerin modelin kararlarını etkilediğinin net bir şekilde görülmesini sağlar. Özellik mühendisliği, veriden doğru ve anlamlı bilgiler çıkarmak için vazgeçilmez bir adımdır ve doğru metriklerle bu özellikleri sıralamak ve seçmek, doğru sonuçlara ulaşmanın en etkili yoludur.

