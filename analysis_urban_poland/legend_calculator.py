from typing import Dict, Tuple, List
import pandas as pd

class LegendCalculator:
    def __init__(self):
        # Przechowuj zakresy dla każdego roku osobno
        self.year_min: Dict[str, Dict[int, float]] = {}  # {data_type: {year: min}}
        self.year_max: Dict[str, Dict[int, float]] = {}  # {data_type: {year: max}}
        
        # Przechowuj wszystkie wartości dla każdego typu danych
        self.history_values: Dict[str, List[float]] = {}  # {data_type: [all_values]}

    def calculate_ranges(self, data: pd.DataFrame, year: int, value_column: str, data_type: str = 'default'):
        """Oblicza zakresy dla danego roku i typu danych."""
        if data_type not in self.year_min:
            self.year_min[data_type] = {}
            self.year_max[data_type] = {}
            self.history_values[data_type] = []
        
        valid_data = data[value_column].dropna()
        if len(valid_data) == 0:
            return
            
        year_min = valid_data.min()
        year_max = valid_data.max()
        
        self.year_min[data_type][year] = year_min
        self.year_max[data_type][year] = year_max
        
        # Dodaj wartości do historii dla tego typu danych
        self.history_values[data_type].extend(valid_data.tolist())

    def get_year_range(self, year: int, data_type: str = 'default') -> Tuple[float, float]:
        """Zwraca zakres dla danego roku i typu danych."""
        if data_type in self.year_min and year in self.year_min[data_type]:
            return self.year_min[data_type][year], self.year_max[data_type][year]
        return 0.0, 1.0

    def get_history_range(self, data_type: str = 'default') -> Tuple[float, float]:
        """Zwraca zakres dla całej historii danego typu danych."""
        if data_type in self.history_values and self.history_values[data_type]:
            values = self.history_values[data_type]
            return float(min(values)), float(max(values))
        return 0.0, 1.0

    def get_history_bins(self, n_bins: int = 7, data_type: str = 'default') -> List[float]:
        """Zwraca granice przedziałów dla legendy uniwersalnej na podstawie wszystkich lat danego typu danych."""
        if data_type not in self.history_values or not self.history_values[data_type]:
            return []
        
        try:
            values = self.history_values[data_type]
            # Usuń duplikaty i posortuj
            unique_values = sorted(set(values))
            
            if len(unique_values) <= n_bins:
                # Jeśli mamy mniej unikalnych wartości niż przedziałów, użyj wszystkich
                return unique_values + [max(unique_values) * 1.01]  # Dodaj górną granicę
            
            # Użyj quantile do stworzenia przedziałów
            return list(pd.qcut(values, n_bins, retbins=True, duplicates='drop')[1])
        except Exception as e:
            print(f"Warning: Failed to create bins for {data_type}: {e}")
            return [] 