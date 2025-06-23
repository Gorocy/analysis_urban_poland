# Analysis Urban Poland

Analiza migracji wewnętrznych i bezrobocia w Polsce na poziomie gmin.

## Funkcjonalności

### Podstawowe analizy
- **Mapy migracji** - wizualizacja saldo migracji na 1000 ludności
- **Mapy bezrobocia** - wizualizacja bezrobocia na 1000 osób
- **Analizy korelacji** - korelacja między migracją a bezrobociem/liczbą ludności
- **Wykresy czasowe** - zmiany korelacji w czasie

### Zaawansowane wizualizacje migracji

#### 1. Flow Maps (Mapy przepływów)
- **Strzałki kierunkowe** pokazujące przepływy migracyjne
- **Napływ** (niebieskie strzałki) - obszary z dodatnim saldo migracji
- **Odpływ** (czerwone strzałki) - obszary z ujemnym saldo migracji
- **Szerokość strzałek** proporcjonalna do wielkości migracji
- Pliki: `mapy/flow_maps/flow_map_YYYY.png`

#### 2. Hot Spot Analysis (Analiza przestrzenna)
- **Hot spots** (czerwone) - obszary o szczególnie wysokim napływie migrantów
- **Cold spots** (niebieskie) - obszary o szczególnie wysokim odpływie migrantów
- **Obszary neutralne** (szare) - obszary bez znaczących zmian
- Próg klasyfikacji: ±1.5 odchylenia standardowego od średniej
- Pliki: `mapy/hotspots/hotspots_YYYY.png`

#### 3. Animacje czasowe
- **Animacja choropleth** - zmiany intensywności migracji w czasie
- **Animacja przepływów** - zmiany kierunków migracji w czasie
- Pokrywa lata 2005-2023
- Pliki: `mapy/animations/migration_*_animation.gif`

#### 4. Mapy porównawcze
- **Przed/po kryzysie finansowym** (2007 vs 2010)
- **Przed/po COVID-19** (2019 vs 2021)
- Pokazują zmiany w strukturze migracji między okresami
- Pliki: `mapy/comparisons/comparison_*_YYYY_YYYY.png`

## Struktura danych

- **data/migracje_wew_i_zew/all.csv** - dane migracji wewnętrznych i zewnętrznych
- **data/bezrobocie/RYNE_1944_CTAB_20250616184618.csv** - dane bezrobocia
- **data/ludnosc/ludnosc.csv** - dane ludności
- **data/map/** - pliki shapefiles z mapami administracyjnymi

## Wyniki

### Foldery wyjściowe
- `mapy/` - podstawowe mapy migracji i bezrobocia
- `mapy/flow_maps/` - mapy przepływów migracyjnych
- `mapy/hotspots/` - mapy hot spotów migracji
- `mapy/animations/` - animacje czasowe
- `mapy/comparisons/` - mapy porównawcze między okresami
- `korelacje/` - wykresy korelacji
- `wyniki_korelacji.csv` - wyniki korelacji w formacie CSV

### Interpretacja wizualizacji

#### Flow Maps
- **Niebieskie strzałki** = napływ migrantów (dodatnie saldo)
- **Czerwone strzałki** = odpływ migrantów (ujemne saldo)
- **Długość strzałki** = wielkość migracji
- **Kierunek strzałki** = kierunek przepływu (do/od centrum regionu)

#### Hot Spots
- **Czerwone obszary** = znacznie ponadprzeciętny napływ
- **Niebieskie obszary** = znacznie ponadprzeciętny odpływ
- **Szare obszary** = migracja w normie
- **Białe obszary** = brak danych

#### Animacje
- **Choropleth** - intensywność kolorów pokazuje wielkość migracji
- **Flow** - ruch strzałek pokazuje zmiany kierunków migracji
- **Rok** wyświetlany w lewym górnym rogu
- **Statystyki** (min/max/średnia) w lewym dolnym rogu

## Uruchomienie

```bash
# Instalacja zależności
poetry install

# Uruchomienie analizy
make run
# lub
poetry run python -m analysis_urban_poland
```

## Wymagania

- Python 3.12+
- pandas - analiza danych
- geopandas - dane przestrzenne
- matplotlib - wizualizacje
- mapclassify - klasyfikacja map

## Lata analizy

- **2005-2014**: Korelacja migracja vs ludność
- **2015-2023**: Korelacja migracja vs bezrobocie
- **Zaawansowane wizualizacje**: Wszystkie dostępne lata

## Uwagi techniczne

- Mapy używają systemu kodów TERYT
- Animacje zapisywane jako GIF (lub MP4 jeśli dostępne)
- Wszystkie wartości migracji normalizowane na 1000 ludności
- Hot spots definiowane jako obszary >1.5σ od średniej