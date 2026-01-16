# Spá um heitavatnsnotkun

Spáðu fyrir um hversu mikið heitt vatn borgin þarf eftir 5 daga!

## Verkefnið

Reykjavík hitar byggingar sínar með jarðhitavatni. Borgin þarf að vita hversu mikið heitt vatn fólk mun nota í framtíðinni til að undirbúa rétt magn.

**Þitt verkefni:** Búðu til líkan sem spáir fyrir um heitavatnsnotkun 120 klukkustundir (5 daga) fram í tímann.

## Inntak og úttak

### Inntak

336 klukkustunda (14 daga) af heitavatnsmælingum:

```
[12543, 12489, 12612, ..., 11089, 11023]
```

### Úttak

Ein tala - spáð notkun eftir 120 klukkustundir:

```
11850
```

## Gögn

| Gagnasett | Tímabil | Tilgangur |
|-----------|---------|-----------|
| Þjálfun | 2020-2023 | Byggja líkan |
| Prófun | 2024 | Prófa í keppni |
| Lokmat | 2025 | Lokaeinkunn |

**Þjálfunargögn:** `data/train.csv`

```python
from utils import load_training_data

history, targets = load_training_data()
```

## Einkunnagjöf

```
Einkunn = max(0, 100 × (1 - villa_þín / grunnlínu_villa))
```

| Einkunn | Merking |
|---------|---------|
| 0 | Sama og grunnlína |
| 50 | Hálfur villufjöldi |
| 100 | Fullkomin spá |

## Fljótleg byrjun

```bash
# Setja upp
pip install -r requirements.txt

# Ræsa þjón
python api.py
```

Þjónninn keyrir á `http://localhost:8080`

## API

```
POST /predict
{
    "history": [12543, 12489, ..., 11023]  // 336 gildi
}

Svar:
{
    "prediction": 11850
}
```

## Ábendingar

1. Byrjaðu einfalt - grunnlínan notar bara gildið frá 5 dögum síðan
2. Skoðaðu mynstur í gögnunum - dagleg og vikuleg sveiflur
3. Nýlegar mælingar eru oft góðar til spár

Gangi þér vel!
