"""Main entrypoint for running the top-level package (if runnable)."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from typing import Any, Dict, List, Optional
from .map_loader import MapLoader
from .data_loader import DataLoader
from .legend_calculator import LegendCalculator
from .map_visualizer import MapVisualizer

def analyze_migration_years():
    # Inicjalizacja komponentów
    map_loader = MapLoader(base_map_path='data/map')
    data_loader = DataLoader(
        migracje_path='data/migracje_wew_i_zew/all.csv',
        bezrobocie_path='data/bezrobocie/RYNE_1944_CTAB_20250616184618.csv',
        ludnosc_path='data/ludnosc/ludnosc.csv'
    )
    legend_calculator = LegendCalculator()
    map_visualizer = MapVisualizer(legend_calculator)

    # Lista do przechowywania wyników korelacji
    correlation_results = []

    # Iteracja po latach
    for year in range(2005, 2024):
        # Wczytanie mapy dla danego roku
        mapa = map_loader.get_map_for_year(year)
        if mapa is None:
            print(f"Nie udało się wczytać mapy dla roku {year}!")
            continue

        print(f"\nRok {year}:")
        print("Przykładowe kody z mapy:", mapa['Kod'].head())
        print("Liczba unikalnych kodów w mapie:", mapa['Kod'].nunique())

        # Pobranie danych dla roku
        migr_data, bezr_data, pop_data = data_loader.get_data_for_year(year)
        if pop_data is None:
            print(f"Brak danych ludności dla roku {year}")
            continue
        
        print("Przykładowe kody z ludności:", pop_data['Kod'].head())
        print("Liczba unikalnych kodów w ludności:", pop_data['Kod'].nunique())
        
        # Dla lat 2005-2014: korelacja migracja vs ludność
        # Dla lat 2015-2023: korelacja migracja vs bezrobocie (jak wcześniej)
        # if year < 2015:
        if year < 2003:
            # Korelacja z liczbą ludności
            if migr_data is not None:
                print("Przykładowe kody z migracji:", migr_data['Kod'].head())
                print("Liczba unikalnych kodów w migracji:", migr_data['Kod'].nunique())
                
                # Znajdź kolumny migracji
                migration_cols = data_loader.get_migration_columns_for_year(year)
                print("Dostępne kolumny migracji:", list(migration_cols.keys()))
                
                # Połącz dane migracji z mapą i ludnością
                mapa_migr = mapa.merge(migr_data, on='Kod', how='left').merge(pop_data[['Kod', 'Ludnosc']], on='Kod', how='left')
                
                # Użyj kolumny 'saldo' i oblicz saldo na 1000 ludności
                if 'saldo' in migration_cols:
                    migr_col = migration_cols['saldo']
                    
                    # Sprawdź czy kolumna została dodana do dataframe i ma dane
                    if migr_col in mapa_migr.columns:
                        # Sprawdź dane migracji przed połączeniem z mapą
                        original_valid_count = migr_data[migr_col].dropna().count()
                        print(f"Oryginalne dane migracji: {original_valid_count} prawidłowych wartości")
                        
                        # Sprawdź dane po połączeniu z mapą
                        merged_valid_count = mapa_migr[migr_col].dropna().count()
                        print(f"Dane po połączeniu z mapą: {merged_valid_count} prawidłowych wartości")
                        
                        if merged_valid_count > 0:
                            print(f"Używam kolumny saldo dla roku {year} ({merged_valid_count} prawidłowych wartości)")
                            
                            # Oblicz saldo migracji na 1000 ludności
                            # Najpierw sprawdź czy mamy dane ludności
                            pop_valid_count = mapa_migr['Ludnosc'].dropna().count()
                            print(f"Dane ludności po połączeniu: {pop_valid_count} prawidłowych wartości")
                            
                            if pop_valid_count > 0:
                                # Oblicz tylko dla rekordów gdzie mamy zarówno migrację jak i ludność
                                mask = mapa_migr[migr_col].notna() & mapa_migr['Ludnosc'].notna() & (mapa_migr['Ludnosc'] > 0)
                                mapa_migr.loc[mask, 'saldo_na_1000'] = (mapa_migr.loc[mask, migr_col] / mapa_migr.loc[mask, 'Ludnosc']) * 1000
                                
                                # Sprawdź czy obliczenie się powiodło
                                valid_saldo_1000 = mapa_migr['saldo_na_1000'].dropna().count()
                                if valid_saldo_1000 > 0:
                                    print(f"Obliczono saldo na 1000 ludności ({valid_saldo_1000} prawidłowych wartości)")
                                    print("Przykładowe wartości saldo na 1000:", mapa_migr['saldo_na_1000'].dropna().head())
                                    
                                    # Sprawdź czy wszystkie wartości są identyczne (problem z universalną skalą)
                                    unique_values = mapa_migr['saldo_na_1000'].dropna().nunique()
                                    if unique_values <= 1:
                                        print(f"Wszystkie wartości saldo na 1000 są identyczne ({unique_values} unikalnych wartości) - pomijam uniwersalną skalę")
                                        skip_universal = True
                                    else:
                                        skip_universal = False
                                    
                                    # Oblicz zakresy dla saldo na 1000
                                    legend_calculator.calculate_ranges(mapa_migr, year, 'saldo_na_1000', 'migracja_saldo_1000')
                                    
                                    # Stwórz folder na mapy
                                    os.makedirs('mapy', exist_ok=True)
                                    
                                    # Twórz mapę z indywidualną skalą
                                    map_visualizer.create_map(
                                        mapa_migr,
                                        year,
                                        'saldo_na_1000',
                                        f'Migration balance per 1000 people {year}',
                                        f'mapy/migracja_saldo_1000_{year}.png',
                                        data_type='migracja_saldo_1000'
                                    )
                                    
                                    # Twórz mapę z uniwersalną skalą tylko jeśli wartości się różnią
                                    if not skip_universal:
                                        map_visualizer.create_map(
                                            mapa_migr,
                                            year,
                                            'saldo_na_1000',
                                            f'Migration balance per 1000 people {year} (universal scale)',
                                            f'mapy/migracja_saldo_1000_{year}_uniwersalna.png',
                                            use_universal_scale=True,
                                            data_type='migracja_saldo_1000'
                                        )
                                    else:
                                        print(f"Pomijam mapę z uniwersalną skalą dla roku {year} - wszystkie wartości identyczne")
                                    
                                    # Oblicz korelację między migracją a liczbą ludności
                                    correlation_data = calculate_correlation_with_population(mapa_migr, 'saldo_na_1000', year, 'saldo_1000')
                                    if correlation_data:
                                        correlation_results.append(correlation_data)
                                else:
                                    print(f"Nie udało się obliczyć saldo na 1000 ludności dla roku {year}")
                            else:
                                print(f"Brak danych ludności po połączeniu dla roku {year}")
                        else:
                            print(f"Nie znaleziono kolumny saldo w dataframe dla roku {year}")
                    else:
                        print(f"Brak kolumny saldo dla roku {year}")
                else:
                    print(f"Brak danych migracji dla roku {year}")
            else:
                print(f"Brak danych migracji dla roku {year}")
        else:
            # Lata 2015-2023: korelacja z bezrobociem (jak wcześniej)
            if bezr_data is None:
                print(f"Brak danych bezrobocia dla roku {year}")
                continue
            
            print("Przykładowe kody z bezrobocia:", bezr_data['Kod'].head())
            print("Liczba unikalnych kodów w bezrobociu:", bezr_data['Kod'].nunique())
            
            # Połączenie danych z mapą
            mapa_bezr = mapa.merge(bezr_data, on='Kod', how='left').merge(pop_data[['Kod', 'Ludnosc']], on='Kod', how='left')
            bezr_col = [c for c in bezr_data.columns if f'{year}' in c and 'osoba' in c][0]
            # Oblicz bezrobocie na 1000 mieszkańców
            mapa_bezr['bezrobocie_na_1000'] = (mapa_bezr[bezr_col] / mapa_bezr['Ludnosc']) * 1000
            print("Przykładowe wartości bezrobocia na 1000:", mapa_bezr['bezrobocie_na_1000'].head())

            # Obliczenie zakresów dla legendy
            legend_calculator.calculate_ranges(mapa_bezr, year, 'bezrobocie_na_1000', 'bezrobocie')

            # Tworzenie map bezrobocia
            os.makedirs('mapy', exist_ok=True)
            map_visualizer.create_map(
                mapa_bezr,
                year,
                'bezrobocie_na_1000',
                f'Bezrobocie na 1000 osób {year}',
                f'mapy/bezrobocie_{year}_na_1000.png',
                data_type='bezrobocie'
            )
            map_visualizer.create_map(
                mapa_bezr,
                year,
                'bezrobocie_na_1000',
                f'Bezrobocie na 1000 osób {year} (skala uniwersalna)',
                f'mapy/bezrobocie_{year}_na_1000_uniwersalna.png',
                use_universal_scale=True,
                data_type='bezrobocie'
            )

            # Tworzenie map migracji i obliczanie korelacji
            if migr_data is not None:
                print("Przykładowe kody z migracji:", migr_data['Kod'].head())
                print("Liczba unikalnych kodów w migracji:", migr_data['Kod'].nunique())
                
                # Znajdź kolumny migracji
                migration_cols = data_loader.get_migration_columns_for_year(year)
                print("Dostępne kolumny migracji:", list(migration_cols.keys()))
                
                # Połącz dane migracji z mapą
                mapa_migr = mapa.merge(migr_data, on='Kod', how='left')
                
                # Twórz mapy tylko dla saldo_1000 (znormalizowane)
                if 'saldo_1000' in migration_cols:
                    migr_col = migration_cols['saldo_1000']
                    migr_type = 'saldo_1000'
                    
                    if migr_col in mapa_migr.columns:
                        print(f"Tworzenie mapy {migr_type} dla roku {year}")
                        
                        # Oblicz zakresy dla tego typu migracji
                        legend_calculator.calculate_ranges(mapa_migr, year, migr_col, f'migracja_{migr_type}')
                        
                        # Twórz mapę z indywidualną skalą
                        map_visualizer.create_map(
                            mapa_migr,
                            year,
                            migr_col,
                            f'{migr_type.title()} {year}',
                            f'mapy/migracja_{migr_type}_{year}.png',
                            data_type=f'migracja_{migr_type}'
                        )
                        
                        # Twórz mapę z uniwersalną skalą
                        map_visualizer.create_map(
                            mapa_migr,
                            year,
                            migr_col,
                            f'{migr_type.title()} {year} (skala uniwersalna)',
                            f'mapy/migracja_{migr_type}_{year}_uniwersalna.png',
                            use_universal_scale=True,
                            data_type=f'migracja_{migr_type}'
                        )
                        
                        # Przeprowadź zaawansowaną analizę przyczynowo-skutkową
                        analysis_data = perform_advanced_migration_unemployment_analysis(
                            mapa_migr, mapa_bezr, migr_col, bezr_col, year, migr_type
                        )
                        if analysis_data:
                            correlation_results.append(analysis_data)
                    else:
                        print(f"Brak kolumny saldo_1000 dla roku {year}")
                else:
                    print(f"Brak kolumny saldo_1000 dla roku {year}")
            else:
                print(f"Brak danych migracji dla roku {year}")

    # Zapisz wyniki analizy do pliku CSV
    if correlation_results:
        df_results = pd.DataFrame(correlation_results)
        df_results.to_csv('wyniki_analizy_przyczynowej.csv', index=False, encoding='utf-8')
        print(f"\n✅ Zapisano wyniki analizy przyczynowej do: wyniki_analizy_przyczynowej.csv")
        
        # Przeprowadź analizę czasową
        perform_temporal_causality_analysis(correlation_results)
        
        # Stwórz podsumowanie analizy
        create_analysis_summary(correlation_results)
    else:
        print("\n❌ Brak wyników do zapisania")

    # Twórz zaawansowane wizualizacje migracji
    print("\n=== Tworzenie zaawansowanych wizualizacji migracji ===")
    # Użyj mapy z najnowszego roku (2023) dla zaawansowanych wizualizacji
    mapa_advanced = map_loader.get_map_for_year(2023)
    if mapa_advanced is not None:
        create_advanced_migration_visualizations(map_visualizer, mapa_advanced, data_loader)
        # Twórz animacje bezrobocia z poprawnymi legendami
        create_unemployment_animations(map_visualizer, mapa_advanced, data_loader)
    else:
        print("Nie udało się wczytać mapy dla zaawansowanych wizualizacji")

    print("\n=== Zakończono tworzenie zaawansowanych wizualizacji ===")
    print("\n📊 PODSUMOWANIE ULEPSZEŃ:")
    print("✅ Strzałki w flow maps zostały powiększone 3x")
    print("✅ Legenda w animacji choropleth została uproszczona")
    print("✅ Dodano szczegółowe flow maps z większą liczbą strzałek")
    print("✅ Zwiększono czytelność wszystkich elementów")
    print(f"\n📁 Sprawdź wyniki w folderach:")
    print("   • mapy/flow_maps/ - mapy przepływów")
    print("   • mapy/hotspots/ - mapy hot spotów") 
    print("   • mapy/animations/ - animacje czasowe")
    print("   • mapy/comparisons/ - mapy porównawcze")

def calculate_correlation_with_population(mapa_migr: Any, migr_col: str, year: int, migr_type: str) -> Optional[Dict[str, Any]]:
    """Oblicza korelację między migracją a liczbą ludności."""
    try:
        # Usuń brakujące wartości
        correlation_data = mapa_migr[['Kod', migr_col, 'Ludnosc']].dropna()
        
        if len(correlation_data) < 10:  # Minimum 10 obserwacji
            print(f"  Zbyt mało danych dla korelacji {migr_type} vs ludność ({len(correlation_data)} obserwacji)")
            return None
        
        # Oblicz korelację
        correlation = correlation_data[migr_col].corr(correlation_data['Ludnosc'])
        
        print(f"  Korelacja {migr_type} vs ludność: {correlation:.3f} (n={len(correlation_data)})")
        
        # Stwórz wykres korelacji
        create_correlation_scatter_population(correlation_data, migr_col, year, migr_type, correlation)
        
        return {
            'year': year,
            'migration_type': migr_type,
            'correlation': correlation,
            'n_observations': len(correlation_data),
            'migration_mean': correlation_data[migr_col].mean(),
            'migration_std': correlation_data[migr_col].std(),
            'population_mean': correlation_data['Ludnosc'].mean(),
            'population_std': correlation_data['Ludnosc'].std(),
            'comparison_variable': 'ludność'
        }
        
    except Exception as e:
        print(f"  Błąd przy obliczaniu korelacji {migr_type} vs ludność: {e}")
        return None

def create_correlation_scatter_population(data: Any, migr_col: str, year: int, migr_type: str, correlation: float) -> None:
    """Tworzy wykres punktowy korelacji z ludnością."""
    plt.figure(figsize=(8, 6))
    plt.scatter(data[migr_col], data['Ludnosc'], alpha=0.6, s=30)
    plt.xlabel(f'{migr_type.title()} (people)', fontsize=12)
    plt.ylabel('Population count', fontsize=12)
    plt.title(f'Correlation {migr_type} vs population {year}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Dodaj informację o korelacji
    plt.text(0.05, 0.95, f'r = {correlation:.3f}\nn = {len(data)}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    os.makedirs('korelacje', exist_ok=True)
    plt.savefig(f'korelacje/korelacja_{migr_type}_ludnosc_{year}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Zapisano wykres korelacji: korelacje/korelacja_{migr_type}_ludnosc_{year}.png")

def calculate_correlation(mapa_migr: Any, mapa_bezr: Any, migr_col: str, bezr_col: str, year: int, migr_type: str) -> Optional[Dict[str, Any]]:
    """Oblicza korelację między migracją a bezrobociem."""
    try:
        # Połącz dane migracji i bezrobocia
        correlation_data = mapa_migr[['Kod', migr_col]].merge(
            mapa_bezr[['Kod', bezr_col]], on='Kod', how='inner'
        )
        
        # Usuń brakujące wartości
        correlation_data = correlation_data.dropna()
        
        if len(correlation_data) < 10:  # Minimum 10 obserwacji
            print(f"  Zbyt mało danych dla korelacji {migr_type} vs bezrobocie ({len(correlation_data)} obserwacji)")
            return None
        
        # Oblicz korelację
        correlation = correlation_data[migr_col].corr(correlation_data[bezr_col])
        
        print(f"  Korelacja {migr_type} vs bezrobocie: {correlation:.3f} (n={len(correlation_data)})")
        
        # Stwórz wykres korelacji
        create_correlation_scatter(correlation_data, migr_col, bezr_col, year, migr_type, correlation)
        
        return {
            'year': year,
            'migration_type': migr_type,
            'correlation': correlation,
            'n_observations': len(correlation_data),
            'migration_mean': correlation_data[migr_col].mean(),
            'migration_std': correlation_data[migr_col].std(),
            'unemployment_mean': correlation_data[bezr_col].mean(),
            'unemployment_std': correlation_data[bezr_col].std(),
            'comparison_variable': 'bezrobocie'
        }
        
    except Exception as e:
        print(f"  Błąd przy obliczaniu korelacji {migr_type} vs bezrobocie: {e}")
        return None

def create_correlation_scatter(data: Any, migr_col: str, bezr_col: str, year: int, migr_type: str, correlation: float) -> None:
    """Tworzy wykres punktowy korelacji."""
    plt.figure(figsize=(8, 6))
    plt.scatter(data[migr_col], data[bezr_col], alpha=0.6, s=30)
    plt.xlabel(f'{migr_type.title()} (people)', fontsize=12)
    plt.ylabel('Unemployment per 1000 people', fontsize=12)
    plt.title(f'Correlation {migr_type} vs unemployment {year}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Dodaj informację o korelacji
    plt.text(0.05, 0.95, f'r = {correlation:.3f}\nn = {len(data)}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    os.makedirs('korelacje', exist_ok=True)
    plt.savefig(f'korelacje/korelacja_{migr_type}_{year}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Zapisano wykres korelacji: korelacje/korelacja_{migr_type}_{year}.png")

def save_correlation_results(correlation_results: List[Dict[str, Any]]) -> None:
    """Zapisuje wyniki korelacji do pliku CSV."""
    df = pd.DataFrame(correlation_results)
    df = df.round(3)
    df.to_csv('wyniki_korelacji.csv', index=False)
    print(f"\nZapisano wyniki korelacji do pliku: wyniki_korelacji.csv")
    print("Podsumowanie korelacji:")
    print(df.groupby('migration_type')['correlation'].describe())

def create_correlation_plots(correlation_results: List[Dict[str, Any]]) -> None:
    """Tworzy wykresy pokazujące zmiany korelacji w czasie."""
    df = pd.DataFrame(correlation_results)
    
    # Wykres korelacji w czasie dla każdego typu migracji
    plt.figure(figsize=(12, 8))
    
    migration_types = df['migration_type'].unique()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, migr_type in enumerate(migration_types):
        type_data = df[df['migration_type'] == migr_type]
        color = colors[i % len(colors)]  # Cykliczne użycie kolorów
        plt.plot(type_data['year'], type_data['correlation'], 
                marker='o', label=migr_type.title(), color=color, linewidth=2)
    
    plt.xlabel('Rok', fontsize=12)
    plt.ylabel('Korelacja z bezrobociem', fontsize=12)
    plt.title('Korelacja między migracjami a bezrobociem w czasie', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('korelacje_w_czasie.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Zapisano wykres korelacji w czasie: korelacje_w_czasie.png")

def create_advanced_migration_visualizations(map_visualizer: MapVisualizer, mapa: Any, data_loader: DataLoader):
    """Tworzy zaawansowane wizualizacje migracji: flow maps, hot spots i animacje."""
    
    # Stwórz foldery na różne typy wizualizacji
    os.makedirs('mapy/flow_maps', exist_ok=True)
    os.makedirs('mapy/hotspots', exist_ok=True)
    os.makedirs('mapy/animations', exist_ok=True)
    
    # Zbierz dane migracji dla wszystkich lat
    migration_data_dict = {}
    years_list = list(range(2005, 2024))
    
    print("Zbieranie danych migracji dla wszystkich lat...")
    for year in years_list:
        migr_data, _, _ = data_loader.get_data_for_year(year)
        if migr_data is not None:
            migration_cols = data_loader.get_migration_columns_for_year(year)
            if 'saldo' in migration_cols:
                migration_data_dict[year] = migr_data
                print(f"  Rok {year}: {len(migr_data)} rekordów")
    
    if not migration_data_dict:
        print("Brak danych migracji do zaawansowanych wizualizacji")
        return
    
    print(f"Zebrano dane dla {len(migration_data_dict)} lat: {list(migration_data_dict.keys())}")
    
    # 1. FLOW MAPS - mapy przepływów dla wybranych lat
    print("\n1. Tworzenie map przepływów (Flow Maps)...")
    sample_years = [2008, 2012, 2016, 2020, 2023]  # Reprezentatywne lata
    
    for year in sample_years:
        if year in migration_data_dict:
            migr_data = migration_data_dict[year]
            migration_cols = data_loader.get_migration_columns_for_year(year)
            
            if 'saldo' in migration_cols:
                print(f"  Tworzenie flow map dla roku {year}...")
                
                # Twórz standardową flow map
                map_visualizer.create_flow_map(
                    mapa,
                    year,
                    f'Migration flows {year}',
                    f'mapy/flow_maps/flow_map_{year}.png',
                    migr_data,
                    migration_cols['saldo'],
                    flow_threshold=0.1
                )
                
                # Twórz również flow map z niższym progiem (więcej strzałek)
                map_visualizer.create_flow_map(
                    mapa,
                    year,
                    f'Migration flows {year} (detailed)',
                    f'mapy/flow_maps/flow_map_detailed_{year}.png',
                    migr_data,
                    migration_cols['saldo'],
                    flow_threshold=0.05  # Niższy próg = więcej strzałek
                )
    
    # 2. HOT SPOT ANALYSIS - analiza hot spotów dla wybranych lat
    print("\n2. Tworzenie map hot spotów...")
    
    for year in sample_years:
        if year in migration_data_dict:
            migr_data = migration_data_dict[year]
            migration_cols = data_loader.get_migration_columns_for_year(year)
            
            if 'saldo' in migration_cols:
                print(f"  Tworzenie hot spot map dla roku {year}...")
                map_visualizer.create_hotspot_map(
                    mapa,
                    year,
                    f'Migration hot spots {year}',
                    f'mapy/hotspots/hotspots_{year}.png',
                    migr_data,
                    migration_cols['saldo'],
                    hotspot_threshold=1.5
                )
    
    # 3. ANIMACJE CZASOWE
    print("\n3. Tworzenie animacji czasowych...")
    
    # Animacja choropleth (mapa kolorowa)
    print("  Tworzenie animacji choropleth...")
    try:
        # Przygotuj dane z kolumną saldo dla wszystkich lat
        animation_data_dict = {}
        for year, migr_data in migration_data_dict.items():
            migration_cols = data_loader.get_migration_columns_for_year(year)
            if 'saldo' in migration_cols:
                # Dodaj kolumnę 'saldo' do danych
                migr_data_copy = migr_data.copy()
                migr_data_copy['saldo'] = migr_data_copy[migration_cols['saldo']]
                animation_data_dict[year] = migr_data_copy
        
        if animation_data_dict:
            map_visualizer.create_time_animation(
                mapa,
                list(animation_data_dict.keys()),
                'Migration changes over time (2005-2023)',
                'mapy/animations/migration_choropleth_animation.gif',
                animation_data_dict,
                'saldo',
                animation_type='choropleth',
                data_type='migration'
            )
    except Exception as e:
        print(f"  Błąd przy tworzeniu animacji choropleth: {e}")
    
    # Animacja flow (przepływy)
    print("  Tworzenie animacji przepływów...")
    try:
        # Sprawdź czy animation_data_dict istnieje i ma dane
        if 'animation_data_dict' in locals() and animation_data_dict:
            map_visualizer.create_time_animation(
                mapa,
                list(animation_data_dict.keys()),
                'Migration flow changes over time (2005-2023)',
                'mapy/animations/migration_flow_animation.gif',
                animation_data_dict,
                'saldo',
                animation_type='flow',
                data_type='migration'
            )
        else:
            print("  Brak danych do animacji przepływów")
    except Exception as e:
        print(f"  Błąd przy tworzeniu animacji przepływów: {e}")
    
    # 4. ANALIZA PORÓWNAWCZA - porównanie różnych okresów
    print("\n4. Tworzenie map porównawczych...")
    
    # Porównanie przed i po kryzysie finansowym (2007 vs 2010)
    if 2007 in migration_data_dict and 2010 in migration_data_dict:
        create_comparison_maps(map_visualizer, mapa, migration_data_dict, data_loader, 
                             2007, 2010, "Before and after financial crisis")
    
    # Porównanie przed i po COVID (2019 vs 2021)
    if 2019 in migration_data_dict and 2021 in migration_data_dict:
        create_comparison_maps(map_visualizer, mapa, migration_data_dict, data_loader, 
                             2019, 2021, "Before and after COVID-19")
    
    print("\n=== Zakończono tworzenie zaawansowanych wizualizacji ===")

def create_unemployment_animations(map_visualizer: MapVisualizer, mapa: Any, data_loader: DataLoader):
    """Tworzy animacje czasowe dla danych bezrobocia."""
    print("\n=== Tworzenie animacji bezrobocia ===")
    
    # Przygotuj dane bezrobocia dla animacji
    unemployment_data_dict = {}
    
    for year in range(2005, 2024):
        print(f"  Ładowanie danych bezrobocia dla roku {year}...")
        try:
            _, bezr_data, _ = data_loader.get_data_for_year(year)
            if bezr_data is not None and len(bezr_data) > 0:
                # Dodaj kolumnę bezrobocie_na_1000 jeśli nie istnieje
                if 'bezrobocie_na_1000' not in bezr_data.columns:
                    if 'bezrobocie' in bezr_data.columns and 'ludnosc' in bezr_data.columns:
                        bezr_data['bezrobocie_na_1000'] = (bezr_data['bezrobocie'] / bezr_data['ludnosc']) * 1000
                    else:
                        print(f"    Brak wymaganych kolumn dla roku {year}")
                        continue
                
                unemployment_data_dict[year] = bezr_data
            else:
                print(f"    Brak danych bezrobocia dla roku {year}")
        except Exception as e:
            print(f"    Błąd przy ładowaniu danych bezrobocia dla roku {year}: {e}")
    
    if not unemployment_data_dict:
        print("  Brak danych bezrobocia do animacji")
        return
    
    print(f"  Przygotowano dane bezrobocia dla {len(unemployment_data_dict)} lat")
    
    # Animacja choropleth bezrobocia
    print("  Tworzenie animacji choropleth bezrobocia...")
    try:
        os.makedirs('mapy/animations', exist_ok=True)
        map_visualizer.create_time_animation(
            mapa,
            list(unemployment_data_dict.keys()),
            'Unemployment changes over time (2005-2023)',
            'mapy/animations/unemployment_choropleth_animation.gif',
            unemployment_data_dict,
            'bezrobocie_na_1000',
            animation_type='choropleth',
            data_type='unemployment'
        )
    except Exception as e:
        print(f"  Błąd przy tworzeniu animacji choropleth bezrobocia: {e}")
    
    print("=== Zakończono tworzenie animacji bezrobocia ===")

def test_animation_legends():
    """Funkcja testowa do sprawdzenia czy legendy w animacjach działają poprawnie."""
    print("\n=== Test legend w animacjach ===")
    
    try:
        from analysis_urban_poland.map_loader import MapLoader
        from analysis_urban_poland.legend_calculator import LegendCalculator
        
        # Inicjalizuj komponenty
        legend_calculator = LegendCalculator()
        map_loader = MapLoader()
        map_visualizer = MapVisualizer(legend_calculator)
        
        # Pobierz mapę dla najnowszego roku
        mapa = map_loader.get_map_for_year(2023)
        if mapa is None:
            print("  Nie udało się wczytać mapy testowej")
            return
        
        # Stwórz testowe dane migracji
        test_migration_data = {}
        for year in [2020, 2021, 2022]:
            # Stwórz losowe dane migracji dla pierwszych 50 gmin
            sample_codes = mapa['Kod'].head(50).tolist()
            np.random.seed(year)  # Dla powtarzalności
            
            test_data = pd.DataFrame({
                'Kod': sample_codes,
                'saldo': np.random.normal(0, 50, len(sample_codes))  # Średnia 0, odchylenie 50
            })
            test_migration_data[year] = test_data
        
        # Stwórz testowe dane bezrobocia
        test_unemployment_data = {}
        for year in [2020, 2021, 2022]:
            sample_codes = mapa['Kod'].head(50).tolist()
            np.random.seed(year + 100)  # Inne ziarno dla bezrobocia
            
            test_data = pd.DataFrame({
                'Kod': sample_codes,
                'bezrobocie_na_1000': np.random.uniform(15, 60, len(sample_codes))  # Od 15 do 60
            })
            test_unemployment_data[year] = test_data
        
        # Test animacji migracji
        print("  Tworzenie testowej animacji migracji...")
        os.makedirs('mapy/test_animations', exist_ok=True)
        
        map_visualizer.create_time_animation(
            mapa,
            [2020, 2021, 2022],
            'TEST: Animacja migracji z poprawioną legendą',
            'mapy/test_animations/test_migration_animation.gif',
            test_migration_data,
            'saldo',
            animation_type='choropleth',
            data_type='migration'
        )
        
        # Test animacji bezrobocia
        print("  Tworzenie testowej animacji bezrobocia...")
        
        map_visualizer.create_time_animation(
            mapa,
            [2020, 2021, 2022],
            'TEST: Animacja bezrobocia z poprawioną legendą',
            'mapy/test_animations/test_unemployment_animation.gif',
            test_unemployment_data,
            'bezrobocie_na_1000',
            animation_type='choropleth',
            data_type='unemployment'
        )
        
        print("✅ Test animacji zakończony pomyślnie!")
        print("📁 Sprawdź wyniki w folderze: mapy/test_animations/")
        print("🔍 Sprawdź czy:")
        print("   • Legenda migracji ma zakres od ujemnych do dodatnich wartości")
        print("   • Legenda bezrobocia ma zakres od małych do dużych wartości (tylko dodatnie)")
        print("   • Znaczniki na legendzie są równomiernie rozłożone")
        print("   • Kolory odpowiadają rzeczywistym wartościom danych")
        
    except Exception as e:
        print(f"❌ Błąd w teście animacji: {e}")
    
    print("=== Koniec testu legend ===")

def create_comparison_maps(map_visualizer: MapVisualizer, mapa: Any, 
                          migration_data_dict: Dict[int, Any], data_loader: DataLoader,
                          year1: int, year2: int, comparison_name: str):
    """Tworzy mapy porównawcze między dwoma latami."""
    print(f"  Porównanie {year1} vs {year2}: {comparison_name}")
    
    # Pobierz dane dla obu lat
    migr_data1 = migration_data_dict[year1]
    migr_data2 = migration_data_dict[year2]
    
    migration_cols1 = data_loader.get_migration_columns_for_year(year1)
    migration_cols2 = data_loader.get_migration_columns_for_year(year2)
    
    if 'saldo' not in migration_cols1 or 'saldo' not in migration_cols2:
        print(f"    Brak danych saldo dla porównania {year1} vs {year2}")
        return
    
    # Połącz dane z mapą
    merged1 = mapa.merge(migr_data1, on='Kod', how='inner')
    merged2 = mapa.merge(migr_data2, on='Kod', how='inner')
    
    # Oblicz różnicę między latami
    common_codes = set(merged1['Kod']) & set(merged2['Kod'])
    if len(common_codes) < 10:
        print(f"    Zbyt mało wspólnych obszarów dla porównania {year1} vs {year2}")
        return
    
    # Stwórz dataframe z różnicami
    comparison_data = []
    for kod in common_codes:
        val1 = merged1[merged1['Kod'] == kod][migration_cols1['saldo']].iloc[0]
        val2 = merged2[merged2['Kod'] == kod][migration_cols2['saldo']].iloc[0]
        
        if pd.notna(val1) and pd.notna(val2):
            # Pobierz geometrię z oryginalnej mapy (nie z merged, żeby uniknąć duplikatów)
            geometry = mapa[mapa['Kod'] == kod]['geometry'].iloc[0]
            comparison_data.append({
                'Kod': kod,
                'roznica_saldo': val2 - val1,
                'geometry': geometry
            })
    
    if len(comparison_data) < 10:
        print(f"    Zbyt mało prawidłowych danych dla porównania {year1} vs {year2}")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_gdf = gpd.GeoDataFrame(comparison_df, geometry='geometry')
    
    # Twórz mapę różnic
    os.makedirs('mapy/comparisons', exist_ok=True)
    
    # Mapa hot spotów różnic
    map_visualizer.create_hotspot_map(
        mapa,
        year2,  # Użyj późniejszego roku jako referencyjnego
        f'Migration changes: {comparison_name} ({year1} → {year2})',
        f'mapy/comparisons/comparison_hotspots_{year1}_{year2}.png',
        comparison_gdf,
        'roznica_saldo',
        hotspot_threshold=1.0
    )
    
    # Mapa flow różnic
    map_visualizer.create_flow_map(
        mapa,
        year2,
        f'Migration flow changes: {comparison_name} ({year1} → {year2})',
        f'mapy/comparisons/comparison_flow_{year1}_{year2}.png',
        comparison_gdf,
        'roznica_saldo',
        flow_threshold=0.2
    )

def perform_advanced_migration_unemployment_analysis(mapa_migr: Any, mapa_bezr: Any, 
                                                   migr_col: str, bezr_col: str, 
                                                   year: int, migr_type: str) -> Optional[Dict[str, Any]]:
    """Przeprowadza zaawansowaną analizę przyczynowo-skutkową między migracją a bezrobociem."""
    try:
        # Połącz dane migracji i bezrobocia
        analysis_data = mapa_migr[['Kod', migr_col]].merge(
            mapa_bezr[['Kod', bezr_col]], on='Kod', how='inner'
        )
        
        # Usuń brakujące wartości
        analysis_data = analysis_data.dropna()
        
        if len(analysis_data) < 20:  # Minimum 20 obserwacji dla wiarygodnej analizy
            print(f"  Zbyt mało danych dla analizy {migr_type} vs bezrobocie ({len(analysis_data)} obserwacji)")
            return None
        
        migration_values = analysis_data[migr_col].values
        unemployment_values = analysis_data[bezr_col].values
        
        # 1. ANALIZA KIERUNKU MIGRACJI W ZALEŻNOŚCI OD BEZROBOCIA
        high_unemployment_threshold = np.percentile(unemployment_values, 75)  # Górny kwartyl
        low_unemployment_threshold = np.percentile(unemployment_values, 25)   # Dolny kwartyl
        
        high_unemployment_mask = unemployment_values >= high_unemployment_threshold
        low_unemployment_mask = unemployment_values <= low_unemployment_threshold
        
        # Średnie saldo migracji w regionach o wysokim i niskim bezrobociu
        avg_migration_high_unemployment = np.mean(migration_values[high_unemployment_mask])
        avg_migration_low_unemployment = np.mean(migration_values[low_unemployment_mask])
        
        # 2. ANALIZA PRZYCIĄGANIA MIGRACYJNEGO
        positive_migration_mask = migration_values > 0  # Regiony z napływem
        negative_migration_mask = migration_values < 0  # Regiony z odpływem
        
        avg_unemployment_positive_migration = np.mean(unemployment_values[positive_migration_mask])
        avg_unemployment_negative_migration = np.mean(unemployment_values[negative_migration_mask])
        
        # 3. KLASYFIKACJA REGIONÓW
        # Regiony atrakcyjne: niskie bezrobocie + dodatnia migracja
        attractive_regions = (unemployment_values <= low_unemployment_threshold) & (migration_values > 0)
        
        # Regiony odpychające: wysokie bezrobocie + ujemna migracja
        repulsive_regions = (unemployment_values >= high_unemployment_threshold) & (migration_values < 0)
        
        # Regiony paradoksalne: wysokie bezrobocie + dodatnia migracja
        paradoxical_regions = (unemployment_values >= high_unemployment_threshold) & (migration_values > 0)
        
        # Regiony stagnacyjne: niskie bezrobocie + ujemna migracja
        stagnant_regions = (unemployment_values <= low_unemployment_threshold) & (migration_values < 0)
        
        # 4. OBLICZ WSKAŹNIKI
        correlation = np.corrcoef(migration_values, unemployment_values)[0, 1]
        
        # Siła odpychania (jak bardzo wysokie bezrobocie koreluje z odpływem)
        repulsion_strength = avg_migration_high_unemployment
        
        # Siła przyciągania (jak bardzo niskie bezrobocie koreluje z napływem)
        attraction_strength = avg_migration_low_unemployment
        
        # Efektywność rynku pracy (różnica między atrakcyjnością a odpychaniem)
        labor_market_efficiency = attraction_strength - repulsion_strength
        
        # 5. STWÓRZ WIZUALIZACJĘ ANALIZY
        create_advanced_analysis_visualization(analysis_data, migr_col, bezr_col, year, migr_type,
                                             high_unemployment_threshold, low_unemployment_threshold,
                                             attractive_regions, repulsive_regions, 
                                             paradoxical_regions, stagnant_regions)
        
        print(f"  === ANALIZA PRZYCZYNOWO-SKUTKOWA {migr_type.upper()} vs BEZROBOCIE {year} ===")
        print(f"  📊 Regiony atrakcyjne (niskie bezrobocie + napływ): {np.sum(attractive_regions)} ({np.sum(attractive_regions)/len(analysis_data)*100:.1f}%)")
        print(f"  📉 Regiony odpychające (wysokie bezrobocie + odpływ): {np.sum(repulsive_regions)} ({np.sum(repulsive_regions)/len(analysis_data)*100:.1f}%)")
        print(f"  ⚠️  Regiony paradoksalne (wysokie bezrobocie + napływ): {np.sum(paradoxical_regions)} ({np.sum(paradoxical_regions)/len(analysis_data)*100:.1f}%)")
        print(f"  🔄 Regiony stagnacyjne (niskie bezrobocie + odpływ): {np.sum(stagnant_regions)} ({np.sum(stagnant_regions)/len(analysis_data)*100:.1f}%)")
        print(f"  💪 Siła przyciągania: {attraction_strength:.2f}")
        print(f"  💨 Siła odpychania: {repulsion_strength:.2f}")
        print(f"  ⚖️  Efektywność rynku pracy: {labor_market_efficiency:.2f}")
        print(f"  🔗 Korelacja: {correlation:.3f}")
        
        return {
            'year': year,
            'migration_type': migr_type,
            'correlation': correlation,
            'n_observations': len(analysis_data),
            'attractive_regions_count': np.sum(attractive_regions),
            'repulsive_regions_count': np.sum(repulsive_regions),
            'paradoxical_regions_count': np.sum(paradoxical_regions),
            'stagnant_regions_count': np.sum(stagnant_regions),
            'attractive_regions_pct': np.sum(attractive_regions)/len(analysis_data)*100,
            'repulsive_regions_pct': np.sum(repulsive_regions)/len(analysis_data)*100,
            'paradoxical_regions_pct': np.sum(paradoxical_regions)/len(analysis_data)*100,
            'stagnant_regions_pct': np.sum(stagnant_regions)/len(analysis_data)*100,
            'attraction_strength': attraction_strength,
            'repulsion_strength': repulsion_strength,
            'labor_market_efficiency': labor_market_efficiency,
            'avg_migration_high_unemployment': avg_migration_high_unemployment,
            'avg_migration_low_unemployment': avg_migration_low_unemployment,
            'avg_unemployment_positive_migration': avg_unemployment_positive_migration,
            'avg_unemployment_negative_migration': avg_unemployment_negative_migration,
            'high_unemployment_threshold': high_unemployment_threshold,
            'low_unemployment_threshold': low_unemployment_threshold,
            'comparison_variable': 'bezrobocie'
        }
        
    except Exception as e:
        print(f"  Błąd przy analizie {migr_type} vs bezrobocie: {e}")
        return None

def create_advanced_analysis_visualization(data: Any, migr_col: str, bezr_col: str, 
                                         year: int, migr_type: str,
                                         high_unemployment_threshold: float, 
                                         low_unemployment_threshold: float,
                                         attractive_regions: Any, repulsive_regions: Any,
                                         paradoxical_regions: Any, stagnant_regions: Any):
    """Tworzy zaawansowaną wizualizację analizy przyczynowo-skutkowej."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    migration_values = data[migr_col].values
    unemployment_values = data[bezr_col].values
    
    # 1. Wykres punktowy z klasyfikacją regionów
    ax1.scatter(unemployment_values[attractive_regions], migration_values[attractive_regions], 
               c='green', alpha=0.7, s=50, label=f'Atrakcyjne (n={np.sum(attractive_regions)})')
    ax1.scatter(unemployment_values[repulsive_regions], migration_values[repulsive_regions], 
               c='red', alpha=0.7, s=50, label=f'Odpychające (n={np.sum(repulsive_regions)})')
    ax1.scatter(unemployment_values[paradoxical_regions], migration_values[paradoxical_regions], 
               c='orange', alpha=0.7, s=50, label=f'Paradoksalne (n={np.sum(paradoxical_regions)})')
    ax1.scatter(unemployment_values[stagnant_regions], migration_values[stagnant_regions], 
               c='gray', alpha=0.7, s=50, label=f'Stagnacyjne (n={np.sum(stagnant_regions)})')
    
    # Dodaj linie progowe
    ax1.axvline(high_unemployment_threshold, color='red', linestyle='--', alpha=0.5, label='Próg wysokiego bezrobocia')
    ax1.axvline(low_unemployment_threshold, color='green', linestyle='--', alpha=0.5, label='Próg niskiego bezrobocia')
    ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    ax1.set_xlabel('Unemployment per 1000 people', fontsize=12)
    ax1.set_ylabel(f'{migr_type.title()} (people)', fontsize=12)
    ax1.set_title(f'Region classification {year}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram bezrobocia z podziałem na regiony
    bins = np.linspace(unemployment_values.min(), unemployment_values.max(), 20)
    ax2.hist(unemployment_values[migration_values > 0], bins=bins, alpha=0.7, 
            label=f'Inflow (n={np.sum(migration_values > 0)})', color='blue')
    ax2.hist(unemployment_values[migration_values < 0], bins=bins, alpha=0.7, 
            label=f'Outflow (n={np.sum(migration_values < 0)})', color='red')
    ax2.set_xlabel('Unemployment per 1000 people', fontsize=12)
    ax2.set_ylabel('Number of regions', fontsize=12)
    ax2.set_title('Unemployment distribution by migration direction', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Średnie wartości w różnych kategoriach
    categories = ['Attractive', 'Repulsive', 'Paradoxical', 'Stagnant']
    migration_means = [
        np.mean(migration_values[attractive_regions]) if np.sum(attractive_regions) > 0 else 0,
        np.mean(migration_values[repulsive_regions]) if np.sum(repulsive_regions) > 0 else 0,
        np.mean(migration_values[paradoxical_regions]) if np.sum(paradoxical_regions) > 0 else 0,
        np.mean(migration_values[stagnant_regions]) if np.sum(stagnant_regions) > 0 else 0
    ]
    unemployment_means = [
        np.mean(unemployment_values[attractive_regions]) if np.sum(attractive_regions) > 0 else 0,
        np.mean(unemployment_values[repulsive_regions]) if np.sum(repulsive_regions) > 0 else 0,
        np.mean(unemployment_values[paradoxical_regions]) if np.sum(paradoxical_regions) > 0 else 0,
        np.mean(unemployment_values[stagnant_regions]) if np.sum(stagnant_regions) > 0 else 0
    ]
    
    colors = ['green', 'red', 'orange', 'gray']
    bars = ax3.bar(categories, migration_means, color=colors, alpha=0.7)
    ax3.set_ylabel(f'Average {migr_type}', fontsize=12)
    ax3.set_title('Average migration by region type', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Dodaj wartości na słupkach
    for bar, value in zip(bars, migration_means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (max(migration_means) - min(migration_means))*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Średnie bezrobocie według typu regionu
    bars2 = ax4.bar(categories, unemployment_means, color=colors, alpha=0.7)
    ax4.set_ylabel('Average unemployment per 1000 people', fontsize=12)
    ax4.set_title('Average unemployment by region type', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Dodaj wartości na słupkach
    for bar, value in zip(bars2, unemployment_means):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (max(unemployment_means) - min(unemployment_means))*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs('analizy_przyczynowe', exist_ok=True)
    plt.savefig(f'analizy_przyczynowe/analiza_przyczynowa_{migr_type}_{year}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Zapisano analizę przyczynową: analizy_przyczynowe/analiza_przyczynowa_{migr_type}_{year}.png")

def perform_temporal_causality_analysis(correlation_results: List[Dict[str, Any]]) -> None:
    """Przeprowadza analizę czasową sprawdzającą związek przyczynowo-skutkowy między bezrobociem a migracją w czasie."""
    df = pd.DataFrame(correlation_results)
    
    print("\n" + "="*80)
    print("🕐 ANALIZA CZASOWA - CO BYŁO PIERWSZE: BEZROBOCIE CZY MIGRACJA?")
    print("="*80)
    
    # Analiza trendów w czasie
    years = sorted(df['year'].unique())
    
    # Grupuj dane według typu migracji
    migration_types = df['migration_type'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analiza czasowa związków przyczynowo-skutkowych', fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # 1. Korelacje w czasie
    ax1 = axes[0, 0]
    for i, migr_type in enumerate(migration_types):
        type_data = df[df['migration_type'] == migr_type].sort_values('year')
        if len(type_data) > 2:  # Tylko jeśli mamy przynajmniej 3 punkty danych
            color = colors[i % len(colors)]
            ax1.plot(type_data['year'], type_data['correlation'], 
                    marker='o', label=migr_type.title(), color=color, linewidth=2)
    
    ax1.set_xlabel('Rok')
    ax1.set_ylabel('Korelacja z bezrobociem')
    ax1.set_title('Korelacja migracja-bezrobocie w czasie')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Efektywność rynku pracy w czasie
    ax2 = axes[0, 1]
    for i, migr_type in enumerate(migration_types):
        type_data = df[df['migration_type'] == migr_type].sort_values('year')
        if len(type_data) > 2 and 'labor_market_efficiency' in type_data.columns:
            color = colors[i % len(colors)]
            ax2.plot(type_data['year'], type_data['labor_market_efficiency'], 
                    marker='s', label=migr_type.title(), color=color, linewidth=2)
    
    ax2.set_xlabel('Rok')
    ax2.set_ylabel('Efektywność rynku pracy')
    ax2.set_title('Efektywność rynku pracy w czasie')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. Regiony atrakcyjne vs odpychające w czasie
    ax3 = axes[1, 0]
    avg_attractive = df.groupby('year')['attractive_regions_pct'].mean()
    avg_repulsive = df.groupby('year')['repulsive_regions_pct'].mean()
    
    ax3.plot(avg_attractive.index, avg_attractive.values, 
            marker='o', label='Regiony atrakcyjne (%)', color='green', linewidth=3)
    ax3.plot(avg_repulsive.index, avg_repulsive.values, 
            marker='s', label='Regiony odpychające (%)', color='red', linewidth=3)
    
    ax3.set_xlabel('Rok')
    ax3.set_ylabel('Procent regionów')
    ax3.set_title('Dynamika regionów atrakcyjnych vs odpychających')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Analiza okresów kryzysowych
    ax4 = axes[1, 1]
    
    # Identyfikuj okresy kryzysowe
    crisis_periods = [
        (2008, 2010, 'Kryzys finansowy', 'red'),
        (2020, 2021, 'COVID-19', 'orange')
    ]
    
    # Średnia korelacja w czasie
    avg_correlation = df.groupby('year')['correlation'].mean()
    ax4.plot(avg_correlation.index, avg_correlation.values, 
            marker='o', color='blue', linewidth=3, label='Średnia korelacja')
    
    # Zaznacz okresy kryzysowe
    for start_year, end_year, label, color in crisis_periods:
        ax4.axvspan(start_year, end_year, alpha=0.2, color=color, label=label)
    
    ax4.set_xlabel('Rok')
    ax4.set_ylabel('Średnia korelacja')
    ax4.set_title('Wpływ kryzysów na związek bezrobocie-migracja')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('analiza_czasowa_przyczynowosci.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Zapisano analizę czasową: analiza_czasowa_przyczynowosci.png")
    
    # ANALIZA STATYSTYCZNA TRENDÓW
    print("\n📈 ANALIZA TRENDÓW CZASOWYCH:")
    
    for migr_type in migration_types:
        type_data = df[df['migration_type'] == migr_type].sort_values('year')
        if len(type_data) >= 5:  # Przynajmniej 5 lat danych
            correlations = type_data['correlation'].values
            years_subset = type_data['year'].values
            
            # Oblicz trend (regresja liniowa)
            if len(correlations) > 1:
                trend_coef = np.polyfit(years_subset, correlations, 1)[0]
                
                print(f"  {migr_type.title()}:")
                print(f"    📊 Trend korelacji: {trend_coef:.4f}/rok")
                
                if abs(trend_coef) > 0.01:
                    if trend_coef > 0:
                        print(f"    📈 Rosnący związek bezrobocie-migracja (wzmacnianie się zależności)")
                    else:
                        print(f"    📉 Słabnący związek bezrobocie-migracja (osłabianie się zależności)")
                else:
                    print(f"    ➡️  Stabilny związek bezrobocie-migracja")
                
                # Analiza okresów przed i po kryzysach
                pre_crisis_data = type_data[type_data['year'] < 2008]
                crisis_data = type_data[(type_data['year'] >= 2008) & (type_data['year'] <= 2010)]
                post_crisis_data = type_data[type_data['year'] > 2010]
                covid_data = type_data[type_data['year'] >= 2020]
                
                if len(pre_crisis_data) > 0 and len(post_crisis_data) > 0:
                    pre_crisis_corr = pre_crisis_data['correlation'].mean()
                    post_crisis_corr = post_crisis_data['correlation'].mean()
                    crisis_impact = post_crisis_corr - pre_crisis_corr
                    
                    print(f"    🏦 Wpływ kryzysu finansowego: {crisis_impact:+.3f}")
                    
                if len(covid_data) > 0 and len(post_crisis_data) > 0:
                    pre_covid_corr = post_crisis_data[post_crisis_data['year'] < 2020]['correlation'].mean()
                    covid_corr = covid_data['correlation'].mean()
                    covid_impact = covid_corr - pre_covid_corr
                    
                    print(f"    🦠 Wpływ COVID-19: {covid_impact:+.3f}")
    
    # WNIOSKI PRZYCZYNOWE
    print("\n🎯 WNIOSKI PRZYCZYNOWE:")
    
    # Sprawdź czy regiony o wysokim bezrobociu mają ujemną migrację
    avg_repulsion = df['repulsion_strength'].mean()
    avg_attraction = df['attraction_strength'].mean()
    
    print(f"  💨 Średnia siła odpychania (wysokie bezrobocie): {avg_repulsion:.2f}")
    print(f"  💪 Średnia siła przyciągania (niskie bezrobocie): {avg_attraction:.2f}")
    
    if avg_repulsion < -1 and avg_attraction > 1:
        print("  ✅ POTWIERDZONO: Regiony z wysokim bezrobociem odpychają mieszkańców")
        print("  ✅ POTWIERDZONO: Regiony z niskim bezrobociem przyciągają mieszkańców")
    elif avg_repulsion < 0 and avg_attraction > 0:
        print("  ⚠️  CZĘŚCIOWO POTWIERDZONO: Słaby efekt odpychania/przyciągania")
    else:
        print("  ❌ NIE POTWIERDZONO: Brak wyraźnego wzorca migracji względem bezrobocia")
    
    # Analiza regionów paradoksalnych
    avg_paradoxical = df['paradoxical_regions_pct'].mean()
    print(f"  🤔 Średnio {avg_paradoxical:.1f}% regionów to regiony paradoksalne")
    print("     (wysokie bezrobocie + napływ ludności)")
    
    if avg_paradoxical > 15:
        print("  ⚠️  UWAGA: Duży odsetek regionów paradoksalnych - inne czynniki niż bezrobocie")
        print("     wpływają na migrację (np. infrastruktura, edukacja, kultura)")
    
    print("\n" + "="*80)

def create_analysis_summary(correlation_results: List[Dict[str, Any]]) -> None:
    """Tworzy podsumowanie analizy."""
    df = pd.DataFrame(correlation_results)
    
    print("\n" + "="*80)
    print("📋 RAPORT KOŃCOWY - ANALIZA PRZYCZYNOWO-SKUTKOWA")
    print("="*80)
    
    # 1. OGÓLNE STATYSTYKI
    print("\n1️⃣ OGÓLNE STATYSTYKI:")
    print(f"   📊 Przeanalizowano {len(df)} przypadków")
    print(f"   📅 Lata: {df['year'].min()}-{df['year'].max()}")
    print(f"   🔄 Typy migracji: {', '.join(df['migration_type'].unique())}")
    print(f"   🎯 Średnia liczba obserwacji na rok: {df['n_observations'].mean():.0f}")
    
    # 2. ANALIZA KORELACJI
    print("\n2️⃣ ANALIZA KORELACJI:")
    avg_correlation = df['correlation'].mean()
    print(f"   📈 Średnia korelacja: {avg_correlation:.3f}")
    
    if avg_correlation < -0.3:
        print("   ✅ SILNA UJEMNA KORELACJA: Wysokie bezrobocie = odpływ ludności")
    elif avg_correlation < -0.1:
        print("   ⚠️  SŁABA UJEMNA KORELACJA: Częściowy wpływ bezrobocia na odpływ")
    elif avg_correlation > 0.1:
        print("   ❓ DODATNIA KORELACJA: Nieoczekiwany wzorzec - wymagane dalsze badania")
    else:
        print("   ➡️  BRAK KORELACJI: Bezrobocie nie wpływa znacząco na migrację")
    
    # 3. KLASYFIKACJA REGIONÓW
    print("\n3️⃣ KLASYFIKACJA REGIONÓW (średnie wartości):")
    print(f"   🟢 Regiony atrakcyjne: {df['attractive_regions_pct'].mean():.1f}%")
    print(f"   🔴 Regiony odpychające: {df['repulsive_regions_pct'].mean():.1f}%")
    print(f"   🟠 Regiony paradoksalne: {df['paradoxical_regions_pct'].mean():.1f}%")
    print(f"   🔘 Regiony stagnacyjne: {df['stagnant_regions_pct'].mean():.1f}%")
    
    # 4. SIŁY RYNKOWE
    print("\n4️⃣ SIŁY RYNKOWE:")
    avg_attraction = df['attraction_strength'].mean()
    avg_repulsion = df['repulsion_strength'].mean()
    avg_efficiency = df['labor_market_efficiency'].mean()
    
    print(f"   💪 Siła przyciągania: {avg_attraction:.2f}")
    print(f"   💨 Siła odpychania: {avg_repulsion:.2f}")
    print(f"   ⚖️  Efektywność rynku pracy: {avg_efficiency:.2f}")
    
    # 5. ODPOWIEDZI NA PYTANIA BADAWCZE
    print("\n5️⃣ ODPOWIEDZI NA PYTANIA BADAWCZE:")
    
    # Pytanie 1: Czy regiony z wyższym bezrobociem mają ujemne saldo migracji?
    if avg_repulsion < -0.5:
        print("   ✅ TAK: Regiony z wysokim bezrobociem mają wyraźnie ujemne saldo migracji")
    elif avg_repulsion < 0:
        print("   ⚠️  CZĘŚCIOWO: Regiony z wysokim bezrobociem mają lekko ujemne saldo migracji")
    else:
        print("   ❌ NIE: Regiony z wysokim bezrobociem nie mają ujemnego salda migracji")
    
    # Pytanie 2: Czy regiony z niskim bezrobociem przyciągają migrantów?
    if avg_attraction > 0.5:
        print("   ✅ TAK: Regiony z niskim bezrobociem wyraźnie przyciągają migrantów")
    elif avg_attraction > 0:
        print("   ⚠️  CZĘŚCIOWO: Regiony z niskim bezrobociem słabo przyciągają migrantów")
    else:
        print("   ❌ NIE: Regiony z niskim bezrobociem nie przyciągają migrantów")
    
    # Pytanie 3: Czy efektywność rynku pracy wpływa na migrację?
    if avg_efficiency > 1:
        print("   ✅ TAK: Rynek pracy jest efektywny - ludzie migrują do regionów z niskim bezrobociem")
    elif avg_efficiency > 0:
        print("   ⚠️  CZĘŚCIOWO: Rynek pracy jest słabo efektywny")
    else:
        print("   ❌ NIE: Rynek pracy jest nieefektywny - migracja nie następuje zgodnie z bezrobociem")
    
    # 6. TRENDY CZASOWE
    print("\n6️⃣ TRENDY CZASOWE:")
    
    # Sprawdź trendy dla każdego typu migracji
    for migr_type in df['migration_type'].unique():
        type_data = df[df['migration_type'] == migr_type].sort_values('year')
        if len(type_data) >= 3:
            correlations = type_data['correlation'].values
            years = type_data['year'].values
            
            if len(correlations) > 1:
                trend = np.polyfit(years, correlations, 1)[0]
                print(f"   📊 {migr_type.title()}: trend {trend:+.4f}/rok")
    
    # 7. REKOMENDACJE
    print("\n7️⃣ REKOMENDACJE POLITYCZNE:")
    
    if avg_repulsion < -0.5 and avg_attraction > 0.5:
        print("   🎯 REKOMENDACJA 1: Polityka zatrudnienia powinna skupić się na regionach")
        print("      o wysokim bezrobociu, aby zmniejszyć odpływ ludności")
        print("   🎯 REKOMENDACJA 2: Wspieranie regionów atrakcyjnych może zwiększyć")
        print("      ich potencjał przyciągania migrantów")
    
    if df['paradoxical_regions_pct'].mean() > 15:
        print("   🎯 REKOMENDACJA 3: Zbadać inne czynniki wpływające na migrację")
        print("      w regionach paradoksalnych (infrastruktura, edukacja, kultura)")
    
    if avg_efficiency < 0:
        print("   🎯 REKOMENDACJA 4: Poprawa przepływu informacji o rynku pracy")
        print("      między regionami może zwiększyć efektywność migracji")
    
    # 8. STWÓRZ WYKRES PODSUMOWUJĄCY
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Raport końcowy - Analiza przyczynowo-skutkowa', fontsize=16, fontweight='bold')
    
    # Wykres 1: Korelacje według typu migracji
    migration_types = df['migration_type'].unique()
    correlations_by_type = [df[df['migration_type'] == mt]['correlation'].mean() for mt in migration_types]
    
    bars1 = ax1.bar(migration_types, correlations_by_type, 
                   color=['red' if c < 0 else 'green' for c in correlations_by_type])
    ax1.set_ylabel('Średnia korelacja')
    ax1.set_title('Korelacja według typu migracji')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Dodaj wartości na słupkach
    for bar, value in zip(bars1, correlations_by_type):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.01,
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # Wykres 2: Siły rynkowe
    forces = ['Przyciąganie', 'Odpychanie', 'Efektywność']
    force_values = [avg_attraction, avg_repulsion, avg_efficiency]
    colors = ['green', 'red', 'blue']
    
    bars2 = ax2.bar(forces, force_values, color=colors, alpha=0.7)
    ax2.set_ylabel('Siła')
    ax2.set_title('Siły rynkowe')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Dodaj wartości na słupkach
    for bar, value in zip(bars2, force_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05 if height >= 0 else height - 0.05,
                f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # Wykres 3: Klasyfikacja regionów
    region_types = ['Atrakcyjne', 'Odpychające', 'Paradoksalne', 'Stagnacyjne']
    region_percentages = [
        df['attractive_regions_pct'].mean(),
        df['repulsive_regions_pct'].mean(),
        df['paradoxical_regions_pct'].mean(),
        df['stagnant_regions_pct'].mean()
    ]
    colors3 = ['green', 'red', 'orange', 'gray']
    
    bars3 = ax3.bar(region_types, region_percentages, color=colors3, alpha=0.7)
    ax3.set_ylabel('Procent regionów')
    ax3.set_title('Klasyfikacja regionów')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Dodaj wartości na słupkach
    for bar, value in zip(bars3, region_percentages):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Wykres 4: Korelacja w czasie
    yearly_correlation = df.groupby('year')['correlation'].mean()
    ax4.plot(yearly_correlation.index, yearly_correlation.values, 
            marker='o', color='blue', linewidth=3, markersize=8)
    ax4.set_xlabel('Rok')
    ax4.set_ylabel('Średnia korelacja')
    ax4.set_title('Korelacja w czasie')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Zaznacz okresy kryzysowe
    ax4.axvspan(2008, 2010, alpha=0.2, color='red', label='Kryzys finansowy')
    ax4.axvspan(2020, 2021, alpha=0.2, color='orange', label='COVID-19')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('raport_koncowy_analiza_przyczynowa.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n📊 Zapisano raport końcowy: raport_koncowy_analiza_przyczynowa.png")
    
    print("\n" + "="*80)
    print("✅ ANALIZA PRZYCZYNOWO-SKUTKOWA ZAKOŃCZONA")
    print("="*80)

if __name__ == "__main__":
    analyze_migration_years()
