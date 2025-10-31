# Kerékpárbérlés Előrejelzés (Python, Scikit-Learn)

Ez a projekt egy **kerékpárbérlés előrejelző modell** felépítését mutatja be a klasszikus *Bike Sharing Dataset* (2011–2012) adatai alapján.  
A cél: megérteni az időjárás, a naptári jellemzők és az idényhatások szerepét a napi bérlésszámok alakulásában, és egy egyszerű, de megbízható **gépi tanulási előrejelző pipeline-t** készíteni Pythonban.

---

## Projekt felépítése

1. **Adatbetöltés és EDA**
   - Fájlok: `day.csv` (napi), `hour.csv` (órás) adatok.
   - Adattisztítás, hiányzó értékek, típusok rendezése.
   - Időbeli mintázatok (hétvége, hónap, napszak) vizsgálata.

2. **Feature Engineering**
   - Ciklikus naptári jellemzők (sin/cos hónap, nap).
   - Interakciós jellemzők az időjárási változók között.
   - One-Hot Encoder a `ColumnTransformer`-ben.

3. **Baseline modell**
   - `HistGradientBoostingRegressor` (Scikit-Learn).
   - Teljesítménymutatók: RMSE, MAE, sMAPE, R2.
   - Baseline RMSE ≈ **1033**, sMAPE ≈ **20.64%**.

4. **Finomított modell és kalibráció**
   - SHAP-alapú jellemző-súlyozás és újratanítás.
   - “Post-hoc” lineáris **BIAS korrekció** a szélsőséges napok javítására.
   - Finomított RMSE ≈ **900**, sMAPE ≈ **18.18%**.

5. **Idősoros validáció**
   - `TimeSeriesSplit` 5-fold validáció.
   - Az eredmények nem teljesen stabilak: az 1. és 4. időszakban a hiba jelentősen magasabb, ami szezonális váltásokra vagy időjárási különbségekre utal.
   - Összességében a modell jól tanul az ismétlődő mintázatokból, de időbeli általánosítása korlátozott.

---

## Technológiák

- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- shap
- Jupyter Notebook

---

## Eredmények

| Modell | RMSE | MAE | sMAPE | R2 |
|--------|------|-----|-------|----|
| Baseline | 1033.45 | 756.38 | 20.64 % | 0.723 |
| Refined (bias-corrected) | **929.44** | **690.12** | **19.12 %** | **0.752** |
| Refined (bias-corrected+LR) | **900.99** | **690.06** | **18.18 %** | **0.780** |

A modell az alsó tartományban (alacsony forgalmú napok) túlbecsülte, a felsőben alulbecsülte a bérlésszámokat.  
Ezt egy **egyszerű lineáris regressziós korrekcióval** (post-processing lépés) sikerült enyhíteni hasonlóan ahhoz, amit Lee és Chen (2024) is [leírnak](https://arxiv.org/html/2405.15950v2) a gépi tanulási regressziók „középre húzó” torzításáról.

---

## Tanulságok

- A *data leakage* elkerülése kulcsfontosságú idősoros adatoknál.  
- Egyetlen “lineáris” korrekció is képes javítani a modell kalibrációját a szélsőségeknél.  
- Tudományos publikációk nagy segítséget tudnak nyújtani a fejlesztés során, főleg kevés tapasztalattal rendelkezőknek
- A SHAP-analízis segít azonosítani a torzítást okozó feature-öket.  
- Időbeli validáció nélkül a modell túlzottan optimistának tűnhet.

---
## Hogyan futtasd

1. Klónozd a repót:
   ```bash
   git clone https://github.com/SasCsanad/Bike-Sharing-Regression
   cd bike-sharing-forecast
   ```
2. Telepítsd a csomagokat:
   ```bash
   pip install -r requirements.txt
   ```
3. Futtasd a notebookot:
   ```bash
   jupyter notebook main.ipynb
   ```

---

## Mit tanultam az elkészítés során

- Hogyan lehet időalapú jellemzőket **ciklikusan kódolni** (sin–cos), hogy a modell felismerje az ismétlődő mintázatokat.  
- Miért fontos az **idősoros validáció**: ha az adatot véletlenszerűen osztjuk, a modell túl optimistának tűnhet.  
- Hogyan lehet a regressziós modellek **bias-át lineáris korrekcióval** (`LinearRegression`) csökkenteni.  
- Az **sMAPE** metrika jobban mutatja a relatív hibát, mint az MAE vagy az RMSE.  
- A szakirodalomból (Lee & Chen 2024; Cho et al. 2020) megtanultam, hogy az ilyen post-processing lépések egyszerűen, de hatékonyan javítják a kalibrációt.  


## Rólam

Ez a projekt gyakorló feladatként készült, illetve hogy demonstráljam a jelenlegi tudásom és érdeklődésem az adattudomány iránt.  
Ha tetszett a mini-projektem lépjünk kapcsolatba!  
📬 [LinkedIn-profilom](https://www.linkedin.com/in/csan%C3%A1d-sas-495362234/)


## Források  
- Aayam B., Keertan B., Zeus L. *Temporal Encoding Strategies for Energy Time Series Prediction.* [PDF](https://arxiv.org/pdf/2503.15456)
- Lee, C., & Chen, J. (2024). *A Systematic Bias of Machine Learning Regression Models and Its Correction.* [PDF](https://arxiv.org/html/2405.15950v2)  
