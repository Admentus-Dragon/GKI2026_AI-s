
# Gullplatan á Þingvöllum (BabyLM - NLP):

Ríkisstjórnin er að undirbúa stafrænt tímahylki sem verður grafið á Þingvöllum. Ykkur hefur verið falið það hlutverk að reiða fram íslenskt mállíkan sem verður skrifað á gullplötu og geymt í tímahylkinu. Þetta mállíkan á að geyma eins mikla íslenska málþekkingu og hægt er svo hún geti verið endurheimt síðar ef framtíðin fer á versta veg. Hins vegar er sá hængur á að gullplatan rúmar aðeins eitt megabæti. Þið verðið að safna textagögnum og kenna litlu líkani eins mikla almenna málþekkingu og þið getið. Líkanið verður síðan metið á leynigögnum úr Risamálheildinni til að kanna íslenskugetu þess. Framtíð íslenskunnar er í ykkar höndum!

**Markmið:** Byggja líkan sem spáir fyrir um næsta bæti í textaröð. Lægri bita-fyrir-bæti (bits-per-byte) þýðir betri þjöppun og hærri stig á stigatöflu.

---

## Fljótleg byrjun

```bash
# 1. Setja upp háðar pakkana
pip install -r requirements.txt

# 2. (Valfrjálst) Þjálfa eigið líkan á þjálfunargögnum
python train_ngram.py --data data/igc_full

# 3. Búa til innsendingarpakka
python create_submission.py

# 4. Staðfesta innsendinguna (mælt með!)
python check_submission.py

# 5. Hlaða upp submission.zip á keppnissíðuna
```

> **Ath:** Keyrðu fyrst `python create_dataset.py` til að sækja þjálfunargögnin frá HuggingFace.

Meðfylgjandi `submission.zip` er byrjunarpakki; við mælum með að þú þjálfir þitt eigið líkan með þeim tólum sem eru í þessu repo — jafnvel einfalt n-gram líkan þjálfað á fleiri gögnum gefur yfirleitt betri niðurstöðu.

---

## Innsendingarsnið

Innsendingin þín þarf að vera `.zip` skrá (hámark 1 MB) með þessari uppbyggingu:

```
submission.zip
├── model.py        # SKYLDUR - verður að innihalda klasanum Model
├── weights.bin     # Valfrjálst - þyngdir líkanins
├── config.json     # Valfrjálst - stillingar
└── ...             # Önnur nauðsynleg skrár
```

### Skyldur viðmót

`model.py` verður að útfæra þetta nákvæma viðmót:

```python
from pathlib import Path

class Model:
	def __init__(self, submission_dir: Path):
		"""
		Hlaða líkaninu hér.

		Args:
			submission_dir: Slóð að útpakkaða innsendingarmöppunni
							sem inniheldur þyngdir, config o.s.frv.
		"""
		pass

	def predict(self, contexts: list[list[int]]) -> list[list[float]]:
		"""
		Spá fyrir um næsta bæti fyrir hverja samhengi-röð.

		Args:
			contexts: Hópur af bætum (bytes). Hvort samhengi er listi af
					  heiltölum (0-255) sem tákna áður gefin bæt.
					  Röðin getur verið af breytilegri lengd (0 til 512 bæti).

		Returns:
			Logits fyrir næsta bætaspá.
			Lögun: [batch_size, 256] - ein lína fyrir hvert samhengi, 256 möguleg bæt.
			Þetta eru hrá logits (munu fara í gegnum softmax fyrir stigagjöf).
		"""
		pass
```

---

## Stigagjöf

Líkanið er metið á því hversu vel það spáir næsta bætinu:

1. Fyrir hverja stöðu í prófunargögnum fær líkanið síðustu 512 bæt sem samhengi
2. Líkanið skilar logits (ónormeraðar líkur) fyrir öll 256 mögulegu bæt
3. Reiknaður er kross-entropy tap: `-log2(softmax(logits)[rétta_bætið])`
4. Lokaniðurstaða = meðal bita-fyrir-bæti yfir allar spár

Lægri er betra!
- ~8 bits/bæti = slembival (jafndreifing)
- ~5 bits/bæti = grunnþjöppun (viðmiðun)
- ~2 bits/bæti = gott málkvarðalíkan
- ~1 bit/bæti = frábært þjöppunarárangur

---

## Takmarkanir

| Takmörkun | Gildi |
|----------:|:-----|
| Hámark zip-stærð | **1 MB** |
| Hámark útpakkað | 50 MB |
| Minni | 4 GB |
| CPU | 2 kjarna (engin GPU) |
| Net | **Ekkert** - alveg lokuð umhverfi |
| Samhengisgluggi | 512 bæti hámark |
| Batch stærð | 1024 samhengi á kall |
| Tímamörk | 10 mínútur samtals |

---

## Tiltækar pakkar

Umhverfið fyrir matið hefur eftirfarandi pakka uppsetta:

| Pakki | Útgáfa | Athugasemd |
|-------|--------|-----------|
| `torch` | 2.9.1 | **CPU-only** - megintólið |
| `transformers` | 4.57.6 | HuggingFace líkanabókasafn |
| `tensorflow-cpu` | 2.20.0 | TensorFlow (CPU) |
| `jax` | 0.8.2 | JAX vistkerfið |
| `flax` | 0.12.2 | JAX vistkerfið |
| `numpy` | 2.4.1 | Töluleg útreikningur |
| `scipy` | 1.17.0 | Vísindalegir útreikningar |
| `safetensors` | 0.7.0 | Hraður þyngdahlöðunarstuðningur |
| `datasets` | 4.5.0 | HuggingFace datasets |
| `pyarrow` | 22.0.0 | Gagnaröðun/seríalísering |

### Þarf annan pakka?

Ef þú þarft pakka sem ekki er uppsettur, spurðu á Discord snemma í viku. Okkur gæti fært kosti á umhverfið, en það tekur tíma til að prófa og endursetja.

**Ekki gera ráð fyrir að við getum bætt við pakka í síðustu stund!**

---

## Gagnasafnið

Þjálfunargögnin eru úr **IGC-2024** (Icelandic Gigaword Corpus), stórt safn íslensks texta.

```bash
# Sækja og undirbúa þjálfunargögn (~2.1M skjöl)
python create_dataset.py
```

Eftir keyrslu þekkirðu `data/igc_full/` sem inniheldur þjálfunarskjölin.

Ath: Staðfesting og prófun eru tekin úr sama IGC gagnasafni.

---

## Staðfesting innsendingar

Keyrðu alltaf þessa staðfestingu áður en þú hleður upp:

```bash
python check_submission.py submission.zip
```

Hún athugar m.a.:
- Stærðarskilyrði (1 MB þjappað, 50 MB útpakkað)
- Rétt ZIP uppbygging og skrár
- `model.py` er til með réttu `Model` klasanum
- `__init__(self, submission_dir)` og `predict()` eru til
- Líkanið er hægt að innstilla og keyra án villa
- Útgangaformi er rétt (listi af 256 logits fyrir hvert samhengi)
- Getur unnið með batch stærðina 1024

Keyrðu þetta áður en þú sendir inn — það grípur mjög mörg algeng mistök.

---

## Þjálfun eigin líkans

Við bjóðum upp á kóða til að þjálfa n-gram líkön. Mælt er með að byrja þar — það er auðvelt og bætir einkunn.

```bash
# Sækja gögn (einungis einu sinni)
python create_dataset.py

# Þjálfa líkan (prófaðu ýmis n gildi)
python train_ngram.py --data data/igc_full --n 2   # bigram
python train_ngram.py --data data/igc_full --n 3   # trigram (mælt)

# Búa til og staðfesta innsendingu
python create_submission.py
python check_submission.py
```

N-gram valkostir:
- `--n 1` = unigram (byggir á alþjóðlegri tíðni bæta)
- `--n 2` = bigram (notar síðasta bætið sem samhengi)
- `--n 3` = trigram (notar síðustu 2 bæt)
- `--n 4+` = hærri röð (meira samhengi, en stærri skrá)

Hærra `n` gefur betri spár en stærri geymsluþörf; notaðu `--min-count` til að pruna sjaldgæfar raðir til að halda pakkastærð undir 1 MB.

---

## Snjallt-ráð

1. **Byrjaðu einfalt** - meðfylgjandi n-gram byrjunarreitur er gott upphaf
2. **Prófaðu staðbundið** - vertu viss um að líkanið hlaðist og keyri áður en þú pakkar
3. **Gættu stærðarinnar** - 1 MB þjappað, 50 MB útpakkað
4. **Engin netaðgangur** - líkanið má ekki sækja þyngdir við keyrslu
5. **CPU aðeins** - stilltu fyrir hraða á CPU frekar en GPU
6. **Batchaðu vel** - þú færð 1024 samhengi í einu, reyndu að vektora útreikninga

---

## Skrár í þessu repo

| Skrá | Lýsing |
|------|--------|
| `README.md` | Aðal README (enskt) |
| `requirements.txt` | Python háðir pakkar |
| `create_dataset.py` | Sækir þjálfunargögn frá HuggingFace |
| `train_ngram.py` | Þjálfunarskript fyrir n-gram |
| `create_submission.py` | Býr til `submission.zip` |
| `check_submission.py` | Staðfestir innsendingu |
| `submission/model.py` | Dæmi um líkan |
| `submission.zip` | Upphafstakkki - unigram grunnlíkan |

---

## Spurningar?

Spyrðu á Discord! Ef þú þarft nýjan pakkauppsetningu skaltu óska um það snemma í viku.

Gangi þér vel!

