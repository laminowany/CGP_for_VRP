import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
# Wczytanie CSV
df = pd.read_csv("metrics/scoreToEpochs.csv")

# Oddzielamy kolumnę epok
epochs = df["epochs"]

# Rysowanie każdej kolumny osobnika
for col in df.columns[1:]:  # pomijamy "epochs"
    plt.plot(epochs, df[col], marker='o', label=col)

plt.xlabel("Liczba epok")
plt.ylabel("Wartość metryki / kosztu")
plt.title("Wyniki poszczególnych osobników w zależności od epok")
plt.legend()
plt.grid(True)
plt.show()