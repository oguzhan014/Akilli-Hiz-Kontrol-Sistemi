import numpy as np
import fuzzyPy as fuzz
import matplotlib.pyplot as plt
import cv2
import pytesseract

# Tesseract yolunu ayarla (Windows için)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Plaka okuma fonksiyonu
def read_plate(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    plate_text = pytesseract.image_to_string(thresh, config='--psm 6')
    return plate_text.strip()

# Üyelik fonksiyonları
x_R = np.arange(0, 91, 1)
x_W = np.arange(0, 11, 1)
x_S = np.arange(0, 151, 1)
x_E = np.arange(0, 21, 1)
x_O = np.arange(0, 101, 1)

R_kotu = fuzz.trapez(x_R, "SOL", [30, 45])
R_normal = fuzz.ucgen(x_R, [30, 45, 60])
R_iyi = fuzz.trapez(x_R, "SAG", [45, 60])

W_kotu = fuzz.ucgen(x_W, [0, 0, 5])
W_normal = fuzz.ucgen(x_W, [0, 5, 10])
W_iyi = fuzz.ucgen(x_W, [5, 10, 10])

S_az = fuzz.ucgen(x_S, [0, 0, 70])
S_ort = fuzz.ucgen(x_S, [0, 70, 130])
S_cok = fuzz.trapez(x_S, "SAG", [70, 130])

E_az = fuzz.ucgen(x_E, [0, 0, 10])
E_ort = fuzz.ucgen(x_E, [0, 10, 20])
E_cok = fuzz.ucgen(x_E, [10, 20, 20])

O_az = fuzz.trapez(x_O, "SOL", [25, 50])
O_ort = fuzz.ucgen(x_O, [25, 50, 85])
O_cok = fuzz.trapez(x_O, "SAG", [50, 85])

# Inputları al
print("Yol viraj düzeyi girin (0-90)")
input_R = int(input())
print("Hava durumu girin (0-10)")
input_W = int(input())
print("Sürücü ortalama hızı girin (0-150)")
input_S = int(input())
print("Kullanıcı deneyimi yılı girin (0-20)")
input_E = int(input())
print("Plaka görüntüsünün yolunu girin (örn: plate.jpg)")
plate_path = input()

# Plaka okuma
plate_number = read_plate(plate_path)
print(f"Tespit edilen plaka: {plate_number}")

# Üyelik değerlerini hesaplama
R_fit_kotu = fuzz.uyelik(x_R, R_kotu, input_R)
R_fit_normal = fuzz.uyelik(x_R, R_normal, input_R)
R_fit_iyi = fuzz.uyelik(x_R, R_iyi, input_R)

W_fit_kotu = fuzz.uyelik(x_W, W_kotu, input_W)
W_fit_normal = fuzz.uyelik(x_W, W_normal, input_W)
W_fit_iyi = fuzz.uyelik(x_W, W_iyi, input_W)

S_fit_az = fuzz.uyelik(x_S, S_az, input_S)
S_fit_ort = fuzz.uyelik(x_S, S_ort, input_S)
S_fit_cok = fuzz.uyelik(x_S, S_cok, input_S)

E_fit_az = fuzz.uyelik(x_E, E_az, input_E)
E_fit_ort = fuzz.uyelik(x_E, E_ort, input_E)
E_fit_cok = fuzz.uyelik(x_E, E_cok, input_E)

# Kural tanımları
rule1 = np.fmin(np.fmin(R_fit_kotu, W_fit_kotu), O_az)
rule2 = np.fmin(np.fmin(R_fit_normal, W_fit_normal), O_ort)
rule3 = np.fmin(np.fmin(R_fit_iyi, W_fit_iyi), O_cok)
rule4 = np.fmin(np.fmax(S_fit_az, E_fit_az), O_az)
rule5 = np.fmin(np.fmax(S_fit_ort, E_fit_ort), O_ort)
rule6 = np.fmin(np.fmax(S_fit_cok, E_fit_cok), O_cok)

out_az = np.fmax(rule1, rule4)
out_ort = np.fmax(rule2, rule5)
out_cok = np.fmax(rule3, rule6)

# Çıkış grafiği
O_zeros = np.zeros_like(x_O)
fig, ax = plt.subplots(figsize=(7, 4))
ax.fill_between(x_O, O_zeros, out_az, facecolor='r', alpha=0.7)
ax.plot(x_O, O_az, 'r', linestyle='--')
ax.fill_between(x_O, O_zeros, out_ort, facecolor='g', alpha=0.7)
ax.plot(x_O, O_ort, 'g', linestyle='--')
ax.fill_between(x_O, O_zeros, out_cok, facecolor='b', alpha=0.7)
ax.plot(x_O, O_cok, 'b', linestyle='--')
ax.set_title('Hız Sınırı Çıkışı')
plt.tight_layout()
plt.savefig('hiz_cikis.png')

# Durulaştırma
mutlak_bulanik_sonuc = np.fmax(out_az, out_ort, out_cok)
durulastirilmis_sonuc = fuzz.durulastir(x_O, mutlak_bulanik_sonuc, 'ağırlık merkezi')
durulastirilmis_sonuc = durulastirilmis_sonuc * 3/2

# Hız sınırı hesaplama
hizSiniri = 100
hizSiniri_dusuk = hizSiniri - (out_az * durulastirilmis_sonuc)
hizSiniri_yuksek = hizSiniri + (out_cok * durulastirilmis_sonuc)
hizSiniri = (hizSiniri_dusuk + hizSiniri_yuksek) / 2

if out_az > out_cok:
    hizSiniri += (out_ort * durulastirilmis_sonuc) / 3
else:
    hizSiniri -= (out_ort * durulastirilmis_sonuc) / 3

degisim = hizSiniri - 100

# Karar verme
cezalandir = input_S > hizSiniri
if cezalandir:
    print(f"Plaka {plate_number} için hız sınırı {hizSiniri:.2f} km/s ihlal edildi. Ceza uygulanacak!")
else:
    print(f"Plaka {plate_number} için hız sınırı {hizSiniri:.2f} km/s içinde. Ceza yok.")
print(f"Değişim oranı: %{float(degisim/100):.2f}")

cv2.destroyAllWindows()
