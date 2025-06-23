import pandas as pd
from typing import Dict, Tuple, Optional

class DataLoader:
    def __init__(self, migracje_path: str, bezrobocie_path: str, ludnosc_path: Optional[str] = None):
        self.migracje = pd.read_csv(migracje_path, sep=';', quotechar='"')
        self.bezrobocie = pd.read_csv(bezrobocie_path, sep=';', quotechar='"')
        self.ludnosc = None
        if ludnosc_path:
            self.ludnosc = pd.read_csv(ludnosc_path, sep=';', quotechar='"')
            self.ludnosc['Kod'] = self.ludnosc['Kod'].astype(str).str.zfill(7)
        
        # Debug info
        # print("\nDane migracji:")
        # print("Kolumny:", self.migracje.columns.tolist())
        # print("Przykładowe wartości Kod:", self.migracje['Kod'].head(10).tolist())
        # print("\nDane bezrobocia:")
        # print("Kolumny:", self.bezrobocie.columns.tolist())
        # print("Przykładowe wartości Kod:", self.bezrobocie['Kod'].head(10).tolist())
        if self.ludnosc is not None:
            print("\nDane ludności:")
            print("Kolumny:", self.ludnosc.columns.tolist())
            print("Przykładowe wartości Kod:", self.ludnosc['Kod'].head(10).tolist())

    def get_population_for_year(self, year: int) -> Optional[pd.DataFrame]:
        if self.ludnosc is None:
            return None
        # Znajdź kolumnę z ludnością dla danego roku
        pop_col = [c for c in self.ludnosc.columns if str(year) in c and 'osoba' in c]
        if not pop_col:
            return None
        pop_data = self.ludnosc[['Kod', 'Nazwa'] + pop_col].copy()
        pop_data = pop_data.rename(columns={pop_col[0]: 'Ludnosc'})
        pop_data['Ludnosc'] = self.safe_convert_to_float(pop_data['Ludnosc'])
        return pop_data

    def get_migration_columns_for_year(self, year: int) -> Dict[str, str]:
        """Zwraca słownik z kolumnami migracji dla danego roku."""
        year_str = str(year)
        migration_cols = {}
        
        print(f"\nSzukam kolumn dla roku {year_str}:")
        print("Wszystkie kolumny zawierające rok:")
        year_columns = [c for c in self.migracje.columns if year_str in c]
        for col in year_columns:
            print(f"  - {col}")
        
        # Saldo migracji - najważniejsze dla analizy
        saldo_cols = [c for c in self.migracje.columns if 'saldo migracji' in c.lower() and year_str in c and 'na 1000' not in c.lower()]
        print(f"Znalezione kolumny 'saldo migracji': {saldo_cols}")
        if saldo_cols:
            migration_cols['saldo'] = saldo_cols[0]
        
        # Zameldowania ogółem
        zameld_cols = [c for c in self.migracje.columns if 'zameldowania ogółem' in c.lower() and year_str in c]
        print(f"Znalezione kolumny 'zameldowania ogółem': {zameld_cols}")
        if zameld_cols:
            migration_cols['zameldowania'] = zameld_cols[0]
        
        # Wymeldowania ogółem
        wymeld_cols = [c for c in self.migracje.columns if 'wymeldowania ogółem' in c.lower() and year_str in c]
        print(f"Znalezione kolumny 'wymeldowania ogółem': {wymeld_cols}")
        if wymeld_cols:
            migration_cols['wymeldowania'] = wymeld_cols[0]
            
        # Saldo na 1000 ludności (znormalizowane)
        saldo_1000_cols = [c for c in self.migracje.columns if 'saldo migracji na 1000 ludności' in c.lower() and year_str in c]
        print(f"Znalezione kolumny 'saldo migracji na 1000 ludności': {saldo_1000_cols}")
        if saldo_1000_cols:
            migration_cols['saldo_1000'] = saldo_1000_cols[0]
        
        print(f"Finalne kolumny znalezione: {migration_cols}")
        return migration_cols

    def safe_convert_to_float(self, series: pd.Series) -> pd.Series:
        """Bezpieczna konwersja serii do float z obsługą różnych formatów."""
        try:
            # Zamień puste stringi na NaN
            series = series.replace('', pd.NA)
            series = series.replace(' ', pd.NA)
            series = series.replace('-', pd.NA)
            
            # Konwertuj do string i zamień przecinki na kropki
            series = series.astype(str).str.replace(',', '.')
            
            # Zamień 'nan' z powrotem na NaN
            series = series.replace('nan', pd.NA)
            
            # Konwertuj do float
            return pd.to_numeric(series, errors='coerce')
        except Exception as e:
            print(f"Błąd konwersji: {e}")
            return pd.Series([pd.NA] * len(series))

    def get_data_for_year(self, year: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Zwraca dane migracji, bezrobocia i ludności dla danego roku."""
        # Znajdź kolumny migracji używając nowej metody
        migration_cols = self.get_migration_columns_for_year(year)
        bezr_col = [c for c in self.bezrobocie.columns if f'{year}' in c]

        migr_data = None
        if migration_cols:
            # Użyj pierwszej dostępnej kolumny migracji (preferuj saldo)
            migr_col_name = migration_cols.get('saldo') or migration_cols.get('zameldowania') or list(migration_cols.values())[0]
            migr_cols = ['Kod', 'Nazwa', migr_col_name] if 'Kod' in self.migracje.columns else [migr_col_name]
            migr_data = self.migracje[migr_cols].copy()
            migr_data['Kod'] = migr_data['Kod'].astype(str).str.zfill(7)
            migr_data[migr_col_name] = self.safe_convert_to_float(migr_data[migr_col_name])
            
            print(f"\nDane migracji dla roku {year}:")
            print("Liczba rekordów przed filtrowaniem:", len(migr_data))
            print("Użyta kolumna migracji:", migr_col_name[:50] + "..." if len(migr_col_name) > 50 else migr_col_name)
            
            valid_data = migr_data[migr_col_name].dropna()
            print(f"Liczba prawidłowych wartości: {len(valid_data)} z {len(migr_data)}")
            if len(valid_data) > 0:
                print("Przykładowe wartości migracji:", valid_data.head(5).tolist())
                print("Statystyki migracji:", valid_data.describe())
            
            # Dodaj wszystkie kolumny migracji do dataframe
            for col_type, col_name in migration_cols.items():
                # Dodaj kolumnę z oryginalną nazwą oraz z prefiksem migr_
                if col_name not in migr_data.columns:
                    migr_data[col_name] = self.safe_convert_to_float(self.migracje[col_name])
                migr_data[f'migr_{col_type}'] = self.safe_convert_to_float(self.migracje[col_name])

        bezr_data = None
        if bezr_col:
            bezr_cols = ['Kod', 'Nazwa'] + bezr_col if 'Kod' in self.bezrobocie.columns else bezr_col
            bezr_data = self.bezrobocie[bezr_cols].copy()
            bezr_data['Kod'] = bezr_data['Kod'].astype(str).str.zfill(7)
            bezr_data[bezr_col[0]] = self.safe_convert_to_float(bezr_data[bezr_col[0]])
            print(f"\nDane bezrobocia dla roku {year}:")
            print("Liczba rekordów przed filtrowaniem:", len(bezr_data))

        pop_data = self.get_population_for_year(year)
        if pop_data is not None:
            print(f"\nDane ludności dla roku {year}:")
            print("Liczba rekordów:", len(pop_data))

        return migr_data, bezr_data, pop_data 