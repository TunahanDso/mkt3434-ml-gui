import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd  # Veri işlemede kullanacağız
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss, hinge_loss

def train_model(model_name, loss_name, impute_method, result_label):

    """
    Seçilen model ve kayıp fonksiyonu ile veriyi eğitir ve sonucu GUI'de gösterir.
    """
    global loaded_data

    if loaded_data is None:
        result_label.config(text="⚠ Önce bir CSV dosyası yüklemelisiniz!")
        return

    try:
        # === Veriyi Ayır ===
        X = loaded_data.iloc[:, :-1]
        y = loaded_data.iloc[:, -1]

        # === Eksik Verileri İşleme ===
        if impute_method == "Mean Imputation":
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        elif impute_method == "Interpolation":
            X = X.interpolate()

        elif impute_method == "Forward Fill":
            X = X.fillna(method='ffill')

        elif impute_method == "Backward Fill":
            X = X.fillna(method='bfill')

        # === Eğitim/Test Ayrımı ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # === Model Seçimi ===
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "SVR":
            model = SVR()
        else:
            result_label.config(text="❌ Desteklenmeyen model seçildi.")
            return

        # Modeli Eğit
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # === Kayıp (Loss) Hesapla ===
        if loss_name == "MSE":
            loss = mean_squared_error(y_test, predictions)
        elif loss_name == "MAE":
            loss = mean_absolute_error(y_test, predictions)
        elif loss_name == "Huber Loss":
            huber = HuberRegressor().fit(X_train, y_train)
            predictions = huber.predict(X_test)
            loss = mean_squared_error(y_test, predictions)
        else:
            result_label.config(text="❌ Geçersiz kayıp fonksiyonu.")
            return

        result_label.config(
            text=f"✅ Model eğitildi!\n📉 {loss_name} değeri: {loss:.4f}"
        )

    except Exception as e:
        result_label.config(text=f"❌ Hata oluştu: {str(e)}")


def train_classification(model_name, loss_name, impute_method, var_smoothing, priors_str, result_label):

    global loaded_data

    if loaded_data is None:
        result_label.config(text="⚠ Önce bir CSV dosyası yükleyin!")
        return

    try:
        # Verileri ayır
        X = loaded_data.iloc[:, :-1]
        y = loaded_data.iloc[:, -1]

        # Etiketler metinse sayıya çevir
        if y.dtype == object or isinstance(y[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)

        # === Eksik verileri temizle ===
        if impute_method == "Mean Imputation":
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        elif impute_method == "Interpolation":
            X = X.interpolate()

        elif impute_method == "Forward Fill":
            X = X.fillna(method='ffill')

        elif impute_method == "Backward Fill":
            X = X.fillna(method='bfill')

        # === GaussianNB modeli oluştur (var_smoothing + priors)
        try:
            priors = eval(priors_str) if priors_str.strip() else None
        except:
            priors = None

        model = GaussianNB(var_smoothing=float(var_smoothing), priors=priors)
        model.fit(X_train, y_train)

        # Eğitim / test bölme
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # === GaussianNB modeli eğit ===
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # === Loss hesapla ===
        if loss_name == "Cross-Entropy":
            loss = log_loss(y_test, y_prob)

        elif loss_name == "Hinge Loss":
            sgd = SGDClassifier(loss="hinge")
            sgd.fit(X_train, y_train)
            y_hinge = sgd.decision_function(X_test)
            loss = hinge_loss(y_test, y_hinge)

        else:
            result_label.config(text="❌ Geçersiz kayıp fonksiyonu.")
            return

        acc = accuracy_score(y_test, y_pred)

        result_label.config(
            text=f"✅ Model eğitildi!\n🎯 Accuracy: {acc:.4f}\n📉 {loss_name}: {loss:.4f}"
        )

    except Exception as e:
        result_label.config(text=f"❌ Hata oluştu: {str(e)}")


loaded_data = None  # CSV yüklendiğinde buraya kaydedilecek

def load_data(label_to_update):
    """
    Dosya seçme işlemi yapar ve seçilen CSV dosyasını yükleyip label’a yazdırır.
    """
    global loaded_data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        loaded_data = pd.read_csv(file_path)
        label_to_update.config(text=f"Yüklenen Dosya: {file_path.split('/')[-1]}")
    else:
        label_to_update.config(text="Dosya yüklenmedi.")

def start_gui():
    root = tk.Tk()
    root.title("MKT3434 ML GUI")
    root.geometry("800x600")

    tab_control = ttk.Notebook(root)
    regression_tab = ttk.Frame(tab_control)
    classification_tab = ttk.Frame(tab_control)

    tab_control.add(regression_tab, text='Regression')
    tab_control.add(classification_tab, text='Classification')
    tab_control.pack(expand=1, fill='both')

    # === REGRESSION SEKME İÇERİĞİ ===
    reg_label = tk.Label(regression_tab, text="📂 Regresyon için verinizi yükleyin:", font=("Arial", 12))
    reg_label.pack(pady=10)

    # Veri yükleme butonu
    upload_button = tk.Button(
        regression_tab,
        text="Veri Yükle (CSV)",
        command=lambda: load_data(uploaded_file_label)
    )
    upload_button.pack(pady=5)

    # Yüklenen dosyanın adını gösterecek etiket
    uploaded_file_label = tk.Label(regression_tab, text="Henüz veri yüklenmedi.")
    uploaded_file_label.pack(pady=5)
    # === Model seçimi dropdown menüsü ===
    tk.Label(regression_tab, text="🧠 Model Seçin:", font=("Arial", 11)).pack(pady=(10, 0))

    # Kullanıcının seçebileceği model isimleri
    model_options = ["Linear Regression", "SVR"]
    selected_model = tk.StringVar()
    selected_model.set(model_options[0])  # Varsayılan olarak ilkini seç
    model_dropdown = ttk.OptionMenu(regression_tab, selected_model, *model_options)
    model_dropdown.pack(pady=5)

    # === Kayıp fonksiyonu (loss function) seçimi ===
    tk.Label(regression_tab, text="📉 Kayıp Fonksiyonu Seçin:", font=("Arial", 11)).pack(pady=(10, 0))

    # Kullanıcının seçebileceği kayıp fonksiyonları
    loss_options = ["MSE", "MAE", "Huber Loss"]
    selected_loss = tk.StringVar()
    selected_loss.set(loss_options[0])
    loss_dropdown = ttk.OptionMenu(regression_tab, selected_loss, *loss_options)
    loss_dropdown.pack(pady=5)
    # === Eksik Veri İşleme Seçimi ===
    tk.Label(regression_tab, text="🧽 Eksik Verileri İşleme Yöntemi:", font=("Arial", 11)).pack(pady=(10, 0))

    imputation_methods = ["Mean Imputation", "Interpolation", "Forward Fill", "Backward Fill"]
    selected_imputation = tk.StringVar()
    selected_imputation.set(imputation_methods[0])  # Varsayılan değer
    imputation_dropdown = ttk.OptionMenu(regression_tab, selected_imputation, *imputation_methods)
    imputation_dropdown.pack(pady=5)

    # === CLASSIFICATION SEKME İÇERİĞİ ===

    class_label = tk.Label(classification_tab, text="📂 Sınıflandırma için verinizi yükleyin:", font=("Arial", 12))
    class_label.pack(pady=10)

    class_upload_button = tk.Button(
        classification_tab,
        text="Veri Yükle (CSV)",
        command=lambda: load_data(class_uploaded_file_label)
    )
    class_upload_button.pack(pady=5)

    class_uploaded_file_label = tk.Label(classification_tab, text="Henüz veri yüklenmedi.")
    class_uploaded_file_label.pack(pady=5)

    # === Model seçimi (şimdilik tek model) ===
    tk.Label(classification_tab, text="🧠 Model Seçin:", font=("Arial", 11)).pack(pady=(10, 0))
    class_model_options = ["GaussianNB"]
    class_selected_model = tk.StringVar()
    class_selected_model.set(class_model_options[0])
    class_model_dropdown = ttk.OptionMenu(classification_tab, class_selected_model, *class_model_options)
    class_model_dropdown.pack(pady=5)

    # === Kayıp fonksiyonu seçimi ===
    tk.Label(classification_tab, text="📉 Kayıp Fonksiyonu Seçin:", font=("Arial", 11)).pack(pady=(10, 0))
    class_loss_options = ["Cross-Entropy", "Hinge Loss"]
    class_selected_loss = tk.StringVar()
    class_selected_loss.set(class_loss_options[0])
    class_loss_dropdown = ttk.OptionMenu(classification_tab, class_selected_loss, *class_loss_options)
    class_loss_dropdown.pack(pady=5)

    # === Eksik veri işleme seçimi ===
    tk.Label(classification_tab, text="🧽 Eksik Verileri İşleme Yöntemi:", font=("Arial", 11)).pack(pady=(10, 0))
    class_imputation_methods = ["Mean Imputation", "Interpolation", "Forward Fill", "Backward Fill"]
    class_selected_imputation = tk.StringVar()
    class_selected_imputation.set(class_imputation_methods[0])
    class_imputation_dropdown = ttk.OptionMenu(classification_tab, class_selected_imputation, *class_imputation_methods)
    class_imputation_dropdown.pack(pady=5)

    # === var_smoothing girişi ===
    tk.Label(classification_tab, text="⚙ var_smoothing değeri girin (örn: 1e-9):", font=("Arial", 11)).pack(pady=(10, 0))
    smoothing_entry = tk.Entry(classification_tab)
    smoothing_entry.insert(0, "1e-9")  # Varsayılan değer
    smoothing_entry.pack(pady=5)

    # === prior olasılıkları girişi ===
    tk.Label(classification_tab, text="📊 Prior olasılıkları (örn: [0.5, 0.5]) ya da boş bırakın:", font=("Arial", 11)).pack(pady=(10, 0))
    priors_entry = tk.Entry(classification_tab)
    priors_entry.insert(0, "")  # Kullanıcı girmezse None sayılacak
    priors_entry.pack(pady=5)

    # === Eğit Butonu ===
    class_train_button = tk.Button(
        classification_tab,
        text="🚀 Modeli Eğit",
        font=("Arial", 11, "bold"),
        bg="#2196F3",
        fg="white",
        command=lambda: train_classification(
            class_selected_model.get(),
            class_selected_loss.get(),
            class_selected_imputation.get(),
            smoothing_entry.get(),
            priors_entry.get(),
            class_result_label
        )

    )
    class_train_button.pack(pady=15)

    # === Sonuç Göstergesi ===
    class_result_label = tk.Label(
        classification_tab,
        text="Sonuç burada görünecek.",
        font=("Arial", 11),
        fg="blue"
    )
    class_result_label.pack(pady=10)



    # === Modeli Eğit Butonu ===
    train_button = tk.Button(
        regression_tab,
        text="🚀 Modeli Eğit",  # Buton üstündeki yazı
        font=("Arial", 11, "bold"),
        bg="#4CAF50",  # Yeşil buton
        fg="white",
        command=lambda: train_model(
            selected_model.get(),
            selected_loss.get(),
            selected_imputation.get(),  # 👈 yeni parametre
            result_label
        )

    )
    train_button.pack(pady=15)

    # === Sonuç Gösterme Alanı ===
    result_label = tk.Label(
        regression_tab,
        text="Sonuç burada görünecek.",  # Varsayılan metin
        font=("Arial", 11),
        fg="green"
    )
    result_label.pack(pady=10)



    root.mainloop()

start_gui()
