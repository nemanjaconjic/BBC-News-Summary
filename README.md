# BBC News Summary

Projekat za ekstraktivnu sumarizaciju BBC novinskih članaka korišćenjem Transformer modela, PyTorch biblioteke, DistilBERT modela i MLflow platforme.

## Opis projekta

Cilj projekta je razvoj sistema za ekstraktivnu sumarizaciju teksta.

Problem sumarizacije predstavljen je kao problem binarne klasifikacije rečenica. Svaka rečenica iz novinskog članka klasifikuje se kao relevantna ili nerelevantna za uključivanje u konačni sažetak.

Prethodno istrenirani DistilBERT model koristi se kao zamrznuti Transformer model za ekstrakciju karakteristika iz rečenica. Dobijeni vektorski prikazi rečenica koriste se kao ulaz u neuronski klasifikator implementiran pomoću biblioteke PyTorch.

Ispitano je pet različitih konfiguracija klasifikatora korišćenjem GroupKFold unakrsne validacije, dok se MLflow koristi za praćenje eksperimenata.

## Skup podataka

U projektu se koristi BBC News Summary skup podataka koji sadrži 2.225 novinskih članaka i njihovih odgovarajućih sažetaka.

Članci su raspoređeni u pet kategorija:

- business
- entertainment
- politics
- sport
- tech

Nakon obrade članaka i izdvajanja pojedinačnih rečenica, konačni skup podataka sadrži 41.420 rečenica.

Svakoj rečenici pridružena je binarna oznaka:

- `0` - rečenica nije izabrana za sažetak
- `1` - rečenica je izabrana za sažetak

## Obrada podataka

Proces pripreme podataka sastoji se od sledećih koraka:

1. Učitavanje BBC članaka i referentnih sažetaka
2. Validacija parova članak-sažetak
3. Čišćenje teksta
4. Podela članaka na pojedinačne rečenice
5. Izračunavanje sličnosti između rečenice i referentnog sažetka
6. Dodeljivanje binarnih oznaka rečenicama
7. Kreiranje konačnog skupa podataka za treniranje

Konačni skup podataka sadrži sledeća polja:

- kategorija
- naziv fajla
- indeks rečenice
- tekst rečenice
- referentni sažetak
- vrednost sličnosti
- oznaka klase

## Transformer ekstrakcija karakteristika

U projektu se koristi prethodno istrenirani model `distilbert-base-uncased`.

DistilBERT se koristi kao zamrznuti Transformer model za ekstrakciju karakteristika, što znači da njegovi parametri nisu dodatno trenirani na BBC skupu podataka.

Svaka rečenica se tokenizuje i prosleđuje kroz DistilBERT. Dobijeni vektorski prikaz rečenice ima 768 dimenzija.

Konačna matrica embedding vektora ima dimenzije:

```text
41420 x 768
```

Embedding vektori se generišu unapred kako bi se izbeglo ponovno izvršavanje DistilBERT modela tokom treniranja različitih konfiguracija klasifikatora.

## Neuronski klasifikator

Embedding vektori dobijeni pomoću DistilBERT modela predstavljaju ulaz u neuronsku mrežu implementiranu pomoću biblioteke PyTorch.

Osnovna arhitektura klasifikatora je:

```text
DistilBERT embedding
        ↓
Linear sloj
        ↓
ReLU
        ↓
Dropout
        ↓
Linear sloj
        ↓
Binarna klasifikacija rečenice
```

Za razliku od zamrznutog DistilBERT modela, parametri ovog klasifikatora se treniraju na BBC News Summary skupu podataka.

Izlaz klasifikatora određuje da li rečenica treba da bude uključena u ekstraktivni sažetak.

## Unakrsna validacija

Za evaluaciju modela koristi se 3-fold GroupKFold unakrsna validacija.

Rečenice se grupišu prema članku iz kojeg potiču. Na ovaj način rečenice iz istog članka ne mogu istovremeno da se pojave u trening i validacionom skupu.

Time se sprečava curenje podataka između trening i validacionih skupova i dobija realnija procena sposobnosti modela da klasifikuje rečenice iz novih članaka.

## Eksperimenti

Ispitano je pet različitih konfiguracija neuronskog klasifikatora.

| Model | Hidden size | Dropout | Learning rate | Batch size |
|---|---:|---:|---:|---:|
| Transformer 1 | 256 | 0.3 | 0.001 | 128 |
| Transformer 2 | 512 | 0.3 | 0.001 | 128 |
| Transformer 3 | 512 | 0.5 | 0.001 | 128 |
| Transformer 4 | 512 | 0.5 | 0.0005 | 128 |
| Transformer 5 | 512 | 0.5 | 0.0005 | 64 |

Svaka konfiguracija evaluirana je kroz tri fold-a, odnosno ukupno je izvršeno 15 finalnih treninga.

Promenom hiperparametara analiziran je njihov uticaj na performanse klasifikatora.

## Rezultati

Prosečni rezultati unakrsne validacije:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Transformer 1 | 0.6653 | 0.6187 | 0.7513 | 0.6785 |
| Transformer 2 | 0.6641 | 0.6150 | 0.7640 | **0.6812** |
| Transformer 3 | 0.6620 | 0.6174 | 0.7406 | 0.6732 |
| Transformer 4 | 0.6658 | 0.6225 | 0.7361 | 0.6743 |
| Transformer 5 | 0.6658 | 0.6225 | 0.7414 | 0.6755 |

Najveću prosečnu F1 vrednost ostvarila je konfiguracija **Transformer 2**, sa rezultatom:

```text
F1 = 0.6812
```

Pored osnovnih klasifikacionih metrika, tokom eksperimenata beleže se vreme treniranja, vreme inferencije i veličina modela.

## Evaluacija modela

Za poređenje svih pet Transformer konfiguracija generišu se:

- poređenje F1 vrednosti
- poređenje Accuracy, Precision, Recall i F1 metrika
- krive učenja

Za najbolju konfiguraciju, Transformer 2, dodatno se generišu:

- matrica konfuzije
- ROC kriva
- Precision-Recall kriva

Generisani grafikoni čuvaju se u direktorijumu:

```text
results/plots/
```

Numerički rezultati eksperimenata čuvaju se u:

```text
results/metrics/
```

Za svaku konfiguraciju čuvaju se rezultati unakrsne validacije, istorija treniranja i predikcije dobijene nad validacionim fold-ovima.

## MLflow praćenje eksperimenata

MLflow se koristi za automatsko praćenje eksperimenata.

Tokom treniranja beleže se:

- hiperparametri modela
- training loss po epohi
- validation loss po epohi
- Accuracy
- Precision
- Recall
- F1
- vreme treniranja
- vreme inferencije
- veličina modela

Finalna evaluacija obuhvata pet konfiguracija i tri fold-a po konfiguraciji.

MLflow korisnički interfejs može se pokrenuti komandom:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Nakon pokretanja dostupan je na adresi:

```text
http://127.0.0.1:5000
```

## Struktura projekta

```text
BBC-News-Summary/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1_eda.ipynb
├── results/
│   ├── metrics/
│   ├── models/
│   └── plots/
├── src/
│   ├── check_labeling.py
│   ├── check_preprocessing.py
│   ├── compare_transformers.py
│   ├── create_embeddings.py
│   ├── embedding_model.py
│   ├── inspect_dataset.py
│   ├── labeling.py
│   ├── plot_transformer_results.py
│   ├── preprocess.py
│   ├── preprocessing.py
│   ├── train_transformer1_mlflow.py
│   ├── train_transformer2_mlflow.py
│   ├── train_transformer3_mlflow.py
│   ├── train_transformer4_mlflow.py
│   ├── train_transformer5_mlflow.py
│   └── validate_dataset.py
├── requirements.txt
└── README.md
```

## Instalacija

Potrebne biblioteke instaliraju se pomoću:

```bash
pip install -r requirements.txt
```

Preporučuje se korišćenje Python virtuelnog okruženja.

## Pokretanje projekta

Prvo se izvršava priprema podataka:

```bash
python src/preprocessing.py
python src/labeling.py
python src/preprocess.py
```

Nakon toga generišu se DistilBERT embedding vektori:

```bash
python src/create_embeddings.py
```

Zatim se mogu pokrenuti eksperimenti:

```bash
python src/train_transformer1_mlflow.py
python src/train_transformer2_mlflow.py
python src/train_transformer3_mlflow.py
python src/train_transformer4_mlflow.py
python src/train_transformer5_mlflow.py
```

Poređenje rezultata svih konfiguracija pokreće se pomoću:

```bash
python src/compare_transformers.py
```

Grafikoni za evaluaciju generišu se pomoću:

```bash
python src/plot_transformer_results.py
```

## Reproduktivnost

Za treniranje se koristi fiksirana vrednost slučajnog semena.

GroupKFold omogućava reproduktivnu podelu podataka na nivou članaka, dok su verzije korišćenih Python biblioteka sačuvane u fajlu `requirements.txt`.

Rezultati eksperimenata i grafikoni čuvaju se u repozitorijumu.

Veliki generisani fajlovi, kao što su istrenirani modeli, obrađeni skupovi podataka, DistilBERT embedding vektori i lokalna MLflow baza podataka, nisu uključeni u Git repozitorijum.