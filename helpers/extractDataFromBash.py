import re

def extract_floats_to_csv_line(s: str) -> str:
    # Szukamy wszystkie liczby zmiennoprzecinkowe (z kropką lub bez)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', s)
    # Zamieniamy je na float i od razu na string
    numbers_str = [str(float(n)) for n in numbers]
    res = str(int(float(numbers_str[0]))) + ','+ ','.join(numbers_str[2::2]) + ','+  numbers_str[-1]
    # Łączymy przecinkami
    return res

# Przykład użycia
text = """
EPOCHS: 100
Osobnik 0: 4.920506000518799
Osobnik 1: 4.8371500968933105
Osobnik 2: 4.9261603355407715
Osobnik 3: 4.860116004943848
Osobnik 4: 4.915332794189453
Osobnik 5: 4.882442951202393
Osobnik 6: 4.9142255783081055
Osobnik 7: 4.805835247039795
Osobnik 8: 4.866073131561279
Osobnik 9: 5.040713310241699
BASELINE score: 4.837982654571533
"""
csv_line = extract_floats_to_csv_line(text)
print(csv_line)  # -> "0.534,0.87,-0.12"