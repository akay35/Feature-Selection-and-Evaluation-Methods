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

----

## Feature Selection and Evaluation Methods
Feature selection and evaluation are fundamental to the success of machine learning models. As stated in the 2018 Harvard Business Review article, "If your data is bad, your machine learning tools are useless," the performance of a model largely depends on the quality of the data. This highlights the critical need for feature selection and appropriate variable engineering, especially in large datasets.

In machine learning models, not all features contribute equally to prediction power. Unnecessary or highly correlated features can reduce model performance and lead to overfitting. Variance Inflation Factor (VIF) helps detect multicollinearity issues and remove irrelevant features, making the model more reliable.

Furthermore, model performance is not only related to data quality but also to the ability to derive meaningful variables from it. Techniques like Recursive Feature Elimination (RFE) help select the most effective features, reducing model complexity and optimizing performance. As AI pioneer Andrew Ng said, "Applied machine learning is basically feature engineering," emphasizing the importance of variable engineering.

Methods like Feature Importance help determine which variables the model relies on the most, improving prediction power and making the decision-making process more understandable.

---

### What is VIF (Variance Inflation Factor)?
VIF measures the degree of multicollinearity (linear dependency) between variables. A high VIF value for a variable indicates that it is strongly correlated with other variables, which may negatively impact model performance.

#### Interpretation of VIF Values:
VIF < 5: No multicollinearity problem.
5 ≤ VIF < 10: Mild multicollinearity; attention is needed.
VIF ≥ 10: High multicollinearity; consider removing the variable or applying transformations.

![image](https://github.com/akay35/Feature-Selection-and-Evaluation-Methods/blob/main/VIF%20Scores.png)

#### When Should VIF Be Calculated?
VIF analysis can be done both before and after building a model, but it is typically performed before model building.

1. Pre-model Analysis:
The goal is to detect multicollinearity among independent variables and optimize the model. This step is essential before building regression models to ensure that the variables are independent. Variables with high VIF can be removed to avoid overfitting and misleading results.

2. Post-model Analysis:
If the model shows low performance or high error rates (e.g., high RMSE or low R²), VIF analysis can be done to detect if multicollinearity is affecting model performance and to make adjustments if necessary.

#### What Can Be Done After VIF Analysis?
If high VIF values are detected, several solutions can be applied:

Remove Variables with High VIF: Remove variables causing multicollinearity.
Apply Transformations: Use transformations like logarithms or scaling to reduce correlations.
PCA (Principal Component Analysis): Reduce the number of independent variables by applying PCA.
Feature Engineering: Re-define or combine variables to create new ones.
You can use the variance_inflation_factor function from the statsmodels.stats.outliers_influence module to calculate VIF.

---

### What is RFE (Recursive Feature Elimination)?
Recursive Feature Elimination (RFE) is a technique used for feature selection. It aims to improve model performance by identifying and eliminating unnecessary or low-importance features. RFE works iteratively to eliminate the least important features and retain the ones that contribute the most to the model's prediction power.

#### How RFE Works:
Model Training: Train a model initially with all features (usually regression or classification).
Feature Importance Ranking: The model computes the importance of each feature.
Eliminate Features: The least important feature(s) are removed.
Repeat: The model is retrained with the remaining features, and the importance is recalculated.
Stop Criterion: This process continues until a predefined number of features remains or the model reaches a certain performance threshold.
The result is a set of features that provide the best model performance.

![image](https://github.com/akay35/Feature-Selection-and-Evaluation-Methods/blob/main/RFE.png)

#### When Should RFE Be Calculated?
RFE is applied before model creation to eliminate unnecessary features and ensure the model generalizes well. Applying RFE post-model is not ideal because the model would have already been trained with all features, including the irrelevant ones.

#### Benefits of RFE:
Simplifies Models: By removing irrelevant features, RFE simplifies the model.
Reduces Computation Time: Fewer features mean less computation.
Reduces Overfitting: Eliminating unnecessary features reduces the risk of overfitting.
Improves Model Generalization: Models trained on important features tend to generalize better.
Improves Interpretability: Fewer, more meaningful features make model outputs easier to interpret.
You can use the RFE method from sklearn.feature_selection for feature selection.

---

### What is Feature Importance?
Feature Importance measures the impact of each feature on the target variable in a machine learning model. It is used to determine which variables are most influential in the model's prediction power.

When Should Feature Importance Be Calculated?
Feature Importance can be assessed both before and after model building.

1. Pre-model Analysis:
To identify and remove irrelevant features before building the model. Reducing feature space before training helps speed up computation and minimize overfitting.

2. Post-model Analysis:
After training, analyze which features contributed the most to the model's prediction power. This helps in optimizing the model and improving interpretability.

![image](https://github.com/akay35/Feature-Selection-and-Evaluation-Methods/blob/main/Feature%20Importance.png)

The graph illustrates the most impactful features on the model's prediction power, with CLTV being the most important feature.

---

### Example Work
In this example, we used a telecom customer churn dataset containing information about 7,043 customers receiving home phone and internet services from a telecom company in California.

Steps Taken:
Preprocessing: Missing values were filled, categorical variables encoded, and scaling applied.
VIF Analysis: Multicollinearity was detected and high VIF variables were removed.
RFE: The recursive feature elimination technique was applied to select the most relevant features for the model.
Feature Importance: After training the model with LightGBM, feature importance analysis was performed to understand the weight each feature carries.
The process improved the model's performance, reduced unnecessary complexity, and helped derive more accurate insights.
