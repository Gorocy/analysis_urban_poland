import geopandas as gpd
import os
from typing import Optional, List

class MapLoader:
    def __init__(self, base_map_path: str):
        self.base_map_path = base_map_path
        self.teryt_candidates = [
            'JPT_KOD_JE', 'jpt_kod_je', 'TERYT', 'Kod', 'kod', 'KOD_GUS', 'KOD',
            'SYM', 'SYM_GMINA', 'ID', 'ID_GMINY', 'Obszar', 'OBSZAR'
        ]
        self.shp_priority = [
            'Obszary.shp',
            'Obszary_1992_region.shp',
            'jednostki_ewidencyjne.shp',
            'Jednostki_ewidencyjne.shp',
            'A05_Granice_jednostek_ewidencyjnych.shp'
        ]

    def get_map_for_year(self, year: int) -> Optional[gpd.GeoDataFrame]:
        """Wczytuje mapę dla danego roku."""
        map_path = self._find_map_path(year)
        if not map_path:
            return None
        
        mapa = gpd.read_file(map_path)
        teryt_col = self._find_teryt_column(mapa)
        if not teryt_col:
            return None
            
        # Przygotuj kod do joinu - zachowujemy format 7-cyfrowy
        mapa['Kod'] = mapa[teryt_col].astype(str).str.replace('_', '', regex=False).str.zfill(7)
        
        # Debug info
        print(f"\nDane mapy dla roku {year}:")
        print("Liczba rekordów:", len(mapa))
        print("Kolumna TERYT używana:", teryt_col)
        print("Przykładowe kody przed konwersją:", mapa[teryt_col].head(10).tolist())
        print("Przykładowe kody po konwersji:", mapa['Kod'].head(10).tolist())
        print("Długości kodów przed konwersją:", [len(str(x)) for x in mapa[teryt_col].head(10)])
        print("Długości kodów po konwersji:", [len(str(x)) for x in mapa['Kod'].head(10)])
        print("Unikalne długości kodów przed konwersją:", sorted(set([len(str(x)) for x in mapa[teryt_col]])))
        print("Unikalne długości kodów po konwersji:", sorted(set([len(str(x)) for x in mapa['Kod']])))
        
        return mapa

    def _find_map_path(self, year: int) -> Optional[str]:
        """Znajduje ścieżkę do pliku mapy dla danego roku."""
        available_years = []
        for d in os.listdir(self.base_map_path):
            if d.startswith('PRG_jednostki_administracyjne_'):
                try:
                    y = int(d.split('_')[-1])
                    available_years.append(y)
                except ValueError:
                    continue
        
        if not available_years:
            return None
        
        # WAŻNE: Dla lat 2005-2014 używamy mapy z 2015, ponieważ mapy z lat 2005-2014 
        # mają inne kody (10-cyfrowe) niż dane migracji/ludności (7-cyfrowe)
        if year < 2015:
            if 2015 in available_years:
                best_year = 2015
                print(f"Używam mapy z roku 2015 dla roku {year} (problem z kodami w starszych mapach)")
            else:
                # Fallback - znajdź najnowszą dostępną mapę
                best_year = max([y for y in available_years if y >= 2015])
                print(f"Używam mapy z roku {best_year} dla roku {year} (brak mapy z 2015)")
        else:
            # Dla lat 2015+ używamy standardowej logiki
            suitable_years = [y for y in available_years if y <= year]
            if not suitable_years:
                return None
            best_year = max(suitable_years)
            
        folder = f'{self.base_map_path}/PRG_jednostki_administracyjne_{best_year}'
        
        for shp_name in self.shp_priority:
            fpath = os.path.join(folder, shp_name)
            if os.path.exists(fpath):
                return fpath
        return None

    def _find_teryt_column(self, df: gpd.GeoDataFrame) -> Optional[str]:
        """Znajduje kolumnę z kodem TERYT."""
        for col in self.teryt_candidates:
            if col in df.columns:
                return col
        return None 