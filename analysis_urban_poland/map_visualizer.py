import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Optional, List, Dict, Tuple
import geopandas as gpd
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os

class MapVisualizer:
    def __init__(self, legend_calculator: Any):
        self.legend_calculator = legend_calculator

    def create_map(self, 
                  data: Any,  # gpd.GeoDataFrame but avoiding linter error
                  year: int, 
                  value_column: str,
                  title: str,
                  output_path: str,
                  use_universal_scale: bool = False,
                  data_type: str = 'default'):
        """Tworzy mapę z odpowiednią skalą."""
        _, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Always start with default values
        bins: Optional[List[float]] = None
        scheme: Optional[str] = None
        k = 5
        
        # Check if we have valid data
        valid_data = data[value_column].dropna()
        if len(valid_data) == 0:
            print(f"Warning: No valid data in column {value_column}")
            # Plot without classification
            data.plot(ax=ax, color='lightgray', edgecolor='black')
            ax.set_title(title)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            return
        
        unique_values = valid_data.nunique()
        
        if use_universal_scale:
            try:
                # Sprawdź czy mamy różne wartości przed użyciem uniwersalnej skali
                if unique_values <= 1:
                    print(f"Skipping universal scale for {data_type} - all values are the same")
                    bins = None
                else:
                    bins = self.legend_calculator.get_history_bins(n_bins=7, data_type=data_type)
                    # Validate bins
                    if bins is not None and len(bins) > 1:
                        # Check if bins are valid numbers and unique
                        try:
                            # Try to convert to float to validate
                            float_bins = [float(b) for b in bins]
                            # Check for uniqueness
                            if len(set(float_bins)) == len(float_bins):
                                k = len(bins) - 1
                                scheme = 'user_defined'
                                print(f"Using universal scale for {data_type}: {len(bins)-1} classes")
                            else:
                                print(f"Bins not unique for {data_type}, falling back to quantiles")
                                bins = None
                        except (ValueError, TypeError):
                            bins = None
                    else:
                        bins = None
            except Exception as e:
                print(f"Warning: Failed to get history bins for {data_type}: {e}")
                bins = None
        
        # If we don't have valid bins, use quantiles
        if bins is None:
            if unique_values > 1:
                k = min(7 if use_universal_scale else 5, unique_values)
                scheme = 'quantiles'
                print(f"Using quantiles for {data_type}: {k} classes")
            else:
                # If all values are the same, don't use classification
                scheme = None
                k = 1
                print(f"All values the same for {data_type}, no classification")
    
        plot_kwargs = {
            'column': value_column,
            'ax': ax,
            'legend': True,
            'cmap': 'OrRd',
            'edgecolor': 'black',
        }
        
        # Only add classification parameters if we have a scheme
        if scheme is not None:
            plot_kwargs['scheme'] = scheme
            if bins is not None and scheme == 'user_defined':
                plot_kwargs['classification_kwds'] = {'bins': bins}
            else:
                plot_kwargs['k'] = k
    
        data.plot(**plot_kwargs)
        
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved map: {output_path}")

    def create_flow_map(self, 
                       data: Any,  # gpd.GeoDataFrame
                       year: int,
                       title: str,
                       output_path: str,
                       migration_data: Any,
                       value_column: str,
                       flow_threshold: float = 0.1):
        """Tworzy mapę z przepływami migracyjnymi między regionami."""
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        
        # Rysuj podstawową mapę
        data.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.7)
        
        # Połącz dane migracji z mapą
        merged_data = data.merge(migration_data, on='Kod', how='inner')
        
        if len(merged_data) == 0:
            print("Brak danych migracji do wyświetlenia przepływów")
            return
        
        # Usuń brakujące wartości
        valid_data = merged_data.dropna(subset=[value_column])
        
        if len(valid_data) < 2:
            print("Zbyt mało danych do utworzenia przepływów")
            return
        
        # Napraw problem z geometrią - ustaw prawidłową kolumnę geometrii
        if 'geometry' not in valid_data.columns:
            if 'geometry_x' in valid_data.columns:
                valid_data = valid_data.rename(columns={'geometry_x': 'geometry'})
            elif 'geometry_y' in valid_data.columns:
                valid_data = valid_data.rename(columns={'geometry_y': 'geometry'})
        
        # Upewnij się, że to jest GeoDataFrame z prawidłową geometrią
        if not isinstance(valid_data, gpd.GeoDataFrame):
            valid_data = gpd.GeoDataFrame(valid_data, geometry='geometry')
        else:
            valid_data = valid_data.set_geometry('geometry')
        
        # Oblicz środki geometryczne
        centroids = valid_data.geometry.centroid
        x_coords = centroids.x.values
        y_coords = centroids.y.values
        values = valid_data[value_column].values
        
        # Normalizuj wartości dla lepszej wizualizacji
        max_abs_value = np.abs(values).max()
        if max_abs_value == 0:
            print("Wszystkie wartości migracji są zerowe")
            return
        
        # Podziel dane na dodatnie i ujemne (napływ vs odpływ)
        positive_mask = values > flow_threshold * max_abs_value
        negative_mask = values < -flow_threshold * max_abs_value
        
        # Rysuj przepływy dla dodatnich wartości (napływ) - strzałki do środka
        if np.any(positive_mask):
            self._draw_convergent_flows(ax, x_coords[positive_mask], y_coords[positive_mask], 
                                      values[positive_mask], max_abs_value, 'blue', 'Inflow')
        
        # Rysuj przepływy dla ujemnych wartości (odpływ) - strzałki na zewnątrz
        if np.any(negative_mask):
            self._draw_divergent_flows(ax, x_coords[negative_mask], y_coords[negative_mask], 
                                     np.abs(values[negative_mask]), max_abs_value, 'red', 'Outflow')
        
        # Dodaj legendę
        self._add_flow_legend(ax, max_abs_value)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved flow map: {output_path}")

    def _draw_convergent_flows(self, ax, x_coords, y_coords, values, max_value, color, label):
        """Rysuje przepływy zbieżne (napływ) - strzałki kierujące się do centrum obszaru."""
        # Znajdź centrum wszystkich punktów
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        for i in range(len(x_coords)):
            # Kierunek do centrum
            dx = center_x - x_coords[i]
            dy = center_y - y_coords[i]
            
            # Normalizuj kierunek
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx = dx / length
                dy = dy / length
            
            # Skaluj długość strzałki według wartości (zwiększone rozmiary)
            arrow_length = (values[i] / max_value) * 1.5  # Maksymalna długość 1.5 stopnia
            
            # Szerokość strzałki zależna od wartości (zwiększona)
            arrow_width = (values[i] / max_value) * 0.05
            
            # Rysuj strzałkę (większe rozmiary)
            ax.arrow(x_coords[i], y_coords[i], 
                    dx * arrow_length, dy * arrow_length,
                    head_width=arrow_width*5, head_length=arrow_length*0.15,
                    fc=color, ec=color, alpha=0.8, linewidth=2)

    def _draw_divergent_flows(self, ax, x_coords, y_coords, values, max_value, color, label):
        """Rysuje przepływy rozbieżne (odpływ) - strzałki kierujące się na zewnątrz."""
        # Znajdź centrum wszystkich punktów
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        for i in range(len(x_coords)):
            # Kierunek od centrum
            dx = x_coords[i] - center_x
            dy = y_coords[i] - center_y
            
            # Normalizuj kierunek
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx = dx / length
                dy = dy / length
            
            # Skaluj długość strzałki według wartości (zwiększone rozmiary)
            arrow_length = (values[i] / max_value) * 1.5
            
            # Szerokość strzałki zależna od wartości (zwiększona)
            arrow_width = (values[i] / max_value) * 0.05
            
            # Rysuj strzałkę (większe rozmiary)
            ax.arrow(x_coords[i], y_coords[i], 
                    dx * arrow_length, dy * arrow_length,
                    head_width=arrow_width*5, head_length=arrow_length*0.15,
                    fc=color, ec=color, alpha=0.8, linewidth=2)

    def _add_flow_legend(self, ax, max_value):
        """Dodaje legendę do mapy przepływów."""
        # Pozycja legendy
        legend_x = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02
        legend_y = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
        
        # Przykładowe strzałki (większe rozmiary)
        ax.arrow(legend_x, legend_y, 0.5, 0, head_width=0.08, head_length=0.08,
                fc='blue', ec='blue', alpha=0.8, linewidth=2)
        ax.text(legend_x + 0.6, legend_y, 'Migrant inflow', fontsize=12, va='center', fontweight='bold')
        
        ax.arrow(legend_x, legend_y - 0.3, 0.5, 0, head_width=0.08, head_length=0.08,
                fc='red', ec='red', alpha=0.8, linewidth=2)
        ax.text(legend_x + 0.6, legend_y - 0.3, 'Migrant outflow', fontsize=12, va='center', fontweight='bold')
        
        # Informacja o skali (przesunięta w dół)
        ax.text(legend_x, legend_y - 0.6, f'Maksymalna wartość: {max_value:.0f}', 
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    def create_hotspot_map(self, 
                          data: Any,  # gpd.GeoDataFrame
                          year: int,
                          title: str,
                          output_path: str,
                          migration_data: Any,
                          value_column: str,
                          hotspot_threshold: float = 1.5):
        """Tworzy mapę hot spotów i cold spotów migracji."""
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        
        # Połącz dane migracji z mapą
        merged_data = data.merge(migration_data, on='Kod', how='inner')
        
        if len(merged_data) == 0:
            print("Brak danych migracji do analizy hot spotów")
            return
        
        # Usuń brakujące wartości
        valid_data = merged_data.dropna(subset=[value_column])
        
        if len(valid_data) < 5:
            print("Zbyt mało danych do analizy hot spotów")
            return
        
        # Napraw problem z geometrią - ustaw prawidłową kolumnę geometrii
        if 'geometry' not in valid_data.columns:
            if 'geometry_x' in valid_data.columns:
                valid_data = valid_data.rename(columns={'geometry_x': 'geometry'})
            elif 'geometry_y' in valid_data.columns:
                valid_data = valid_data.rename(columns={'geometry_y': 'geometry'})
        
        # Upewnij się, że to jest GeoDataFrame z prawidłową geometrią
        if not isinstance(valid_data, gpd.GeoDataFrame):
            valid_data = gpd.GeoDataFrame(valid_data, geometry='geometry')
        else:
            valid_data = valid_data.set_geometry('geometry')
        
        values = valid_data[value_column].values
        
        # Oblicz statystyki
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Klasyfikuj obszary
        hot_spots = values > (mean_val + hotspot_threshold * std_val)
        cold_spots = values < (mean_val - hotspot_threshold * std_val)
        neutral = ~(hot_spots | cold_spots)
        
        # Rysuj obszary z różnymi kolorami
        if np.any(hot_spots):
            valid_data[hot_spots].plot(ax=ax, color='red', alpha=0.8, edgecolor='darkred', 
                                     linewidth=1, label=f'Hot spots (n={np.sum(hot_spots)})')
        
        if np.any(cold_spots):
            valid_data[cold_spots].plot(ax=ax, color='blue', alpha=0.8, edgecolor='darkblue', 
                                      linewidth=1, label=f'Cold spots (n={np.sum(cold_spots)})')
        
        if np.any(neutral):
            valid_data[neutral].plot(ax=ax, color='lightgray', alpha=0.6, edgecolor='gray', 
                                   linewidth=0.5, label=f'Neutralne (n={np.sum(neutral)})')
        
        # Dodaj obszary bez danych
        missing_data = data[~data['Kod'].isin(valid_data['Kod'])]
        if len(missing_data) > 0:
            missing_data.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5, 
                            label=f'Brak danych (n={len(missing_data)})')
        
        # Dodaj etykiety dla najważniejszych hot spotów i cold spotów
        self._add_hotspot_labels(ax, valid_data, values, hot_spots, cold_spots, value_column)
        
        # Dodaj legendę
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        # Dodaj informacje statystyczne
        stats_text = f'Mean: {mean_val:.1f}\nStd dev: {std_val:.1f}\nThreshold: ±{hotspot_threshold:.1f}σ'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved hotspot map: {output_path}")

    def _add_hotspot_labels(self, ax, data, values, hot_spots, cold_spots, value_column):
        """Dodaje etykiety do najważniejszych hot spotów i cold spotów."""
        # Znajdź top 3 hot spoty
        if np.any(hot_spots):
            hot_indices = np.where(hot_spots)[0]
            top_hot = hot_indices[np.argsort(values[hot_indices])[-3:]]
            
            for idx in top_hot:
                centroid = data.iloc[idx].geometry.centroid
                ax.annotate(f'{values[idx]:.0f}', 
                          xy=(centroid.x, centroid.y), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, fontweight='bold', color='darkred',
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Znajdź top 3 cold spoty
        if np.any(cold_spots):
            cold_indices = np.where(cold_spots)[0]
            top_cold = cold_indices[np.argsort(values[cold_indices])[:3]]
            
            for idx in top_cold:
                centroid = data.iloc[idx].geometry.centroid
                ax.annotate(f'{values[idx]:.0f}', 
                          xy=(centroid.x, centroid.y), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, fontweight='bold', color='darkblue',
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    def create_time_animation(self, 
                            data: Any,  # gpd.GeoDataFrame
                            years: List[int],
                            title: str,
                            output_path: str,
                            migration_data_dict: Dict[int, Any],
                            value_column: str,
                            animation_type: str = 'choropleth',
                            data_type: str = 'migration'):
        """Tworzy animację czasową pokazującą zmiany migracji."""
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        
        # Przygotuj dane dla wszystkich lat
        yearly_data = {}
        all_values = []
        
        for year in years:
            if year not in migration_data_dict:
                continue
                
            migr_data = migration_data_dict[year]
            merged_data = data.merge(migr_data, on='Kod', how='inner')
            
            if len(merged_data) == 0 or value_column not in merged_data.columns:
                continue
            
            valid_data = merged_data.dropna(subset=[value_column])
            if len(valid_data) > 0:
                # Napraw problem z geometrią
                if 'geometry' not in valid_data.columns:
                    if 'geometry_x' in valid_data.columns:
                        valid_data = valid_data.rename(columns={'geometry_x': 'geometry'})
                    elif 'geometry_y' in valid_data.columns:
                        valid_data = valid_data.rename(columns={'geometry_y': 'geometry'})
                
                # Upewnij się, że to jest GeoDataFrame z prawidłową geometrią
                if not isinstance(valid_data, gpd.GeoDataFrame):
                    valid_data = gpd.GeoDataFrame(valid_data, geometry='geometry')
                else:
                    valid_data = valid_data.set_geometry('geometry')
                
                yearly_data[year] = valid_data
                all_values.extend(valid_data[value_column].values)
        
        if not yearly_data:
            print("Brak danych do animacji")
            return
        
        # Oblicz globalny zakres wartości dla spójnej skali
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        
        # Rozszerz zakres o 5% dla lepszej wizualizacji
        value_range = max_val - min_val
        padding = value_range * 0.05
        min_val -= padding
        max_val += padding
        
        print(f"  Zakres wartości animacji: {min_val:.2f} do {max_val:.2f}")
        print(f"  Liczba lat z danymi: {len(yearly_data)}")
        
        # Inicjalizuj mapę
        if animation_type == 'choropleth':
            im = self._init_choropleth_animation(ax, data, min_val, max_val)
        else:  # flow
            im = None
        
        # Dodaj tekst z rokiem
        year_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=16, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                           verticalalignment='top')
        
        # Dodaj statystyki
        stats_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           verticalalignment='top')
        
        def animate(frame):
            ax.clear()
            year = years[frame % len(years)]
            
            if year not in yearly_data:
                # Pokaż pustą mapę
                data.plot(ax=ax, color='lightgray', edgecolor='black')
                year_text.set_text(f'Year: {year} (no data)')
                stats_text.set_text('')
            else:
                year_data = yearly_data[year]
                values = year_data[value_column].values
                
                if animation_type == 'choropleth':
                    # Wybierz odpowiednią mapę kolorów
                    if data_type == 'unemployment':
                        cmap = 'Reds'
                    elif data_type == 'migration':
                        cmap = 'RdBu_r'
                    else:
                        cmap = 'viridis'
                    
                    # Mapa choropleth z ustaloną skalą kolorów dla wszystkich klatek
                    year_data.plot(column=value_column, ax=ax, cmap=cmap, 
                                 vmin=min_val, vmax=max_val, legend=False,
                                 edgecolor='black', linewidth=0.5)
                    
                    # Upewnij się, że wszystkie wartości są w zakresie min_val-max_val
                    # (przytnij wartości wykraczające poza zakres)
                    current_values = year_data[value_column].values
                    clipped_values = np.clip(current_values, min_val, max_val)
                    
                    # Dodaj obszary bez danych
                    missing_data = data[~data['Kod'].isin(year_data['Kod'])]
                    if len(missing_data) > 0:
                        missing_data.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5)
                    
                    # Dodaj jedną stałą legendę kolorów (tylko w pierwszej klatce)
                    if frame == 0:
                        # Stwórz uniwersalną legendę kolorów z równomiernym podziałem
                        from matplotlib.colors import Normalize
                        from matplotlib.cm import ScalarMappable
                        import matplotlib.ticker as ticker
                        
                        norm = Normalize(vmin=min_val, vmax=max_val)
                        
                        # Wybierz odpowiednią mapę kolorów i etykietę w zależności od typu danych
                        if data_type == 'unemployment':
                            cmap = 'Reds'
                            label = 'Unemployment per 1000 people'
                        elif data_type == 'migration':
                            cmap = 'RdBu_r'
                            label = 'Migration balance (people)'
                        else:
                            cmap = 'viridis'
                            label = value_column
                        
                        sm = ScalarMappable(norm=norm, cmap=cmap)
                        sm.set_array([])
                        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
                        cbar.set_label(label, rotation=270, labelpad=15)
                        
                        # Ustaw równomierne znaczniki na legendzie
                        # Oblicz rozsądną liczbę znaczników (5-8 dla lepszej czytelności)
                        value_range = max_val - min_val
                        if value_range > 10000:
                            # Dla dużych wartości, użyj kroków co 500, 1000, itd.
                            if value_range > 100000:
                                step = 1000
                            elif value_range > 50000:
                                step = 500
                            else:
                                step = 2500
                        elif value_range > 1000:
                            # Dla średnich wartości, użyj kroków co 50, 100, itd.
                            if value_range > 500:
                                step = 100
                            else:
                                step = 50
                        elif value_range > 10:
                            # Dla małych wartości, użyj kroków co 5, 10, itd.
                            if value_range > 50:
                                step = 10
                            else:
                                step = 5
                        else:
                            # Dla bardzo małych wartości, użyj kroków co 1, 2, itd.
                            step = max(1, int(value_range / 5))
                        
                        # Stwórz równomierne znaczniki oparte na rzeczywistych danych
                        original_min = np.min(all_values)
                        original_max = np.max(all_values)
                        
                        # Użyj rzeczywistego zakresu danych do znaczników
                        data_range = original_max - original_min
                        
                        # Oblicz rozsądny krok dla znaczników
                        if data_range > 1000:
                            step = 1000 if data_range > 5000 else 500
                        elif data_range > 100:
                            step = 100 if data_range > 500 else 50
                        elif data_range > 10:
                            step = 10 if data_range > 50 else 5
                        else:
                            step = max(1, int(data_range / 5))
                        
                        # Znajdź pierwszą wartość znacznika (zaokrąglona w dół)
                        first_tick = int(original_min // step) * step
                        if first_tick < original_min:
                            first_tick += step
                        
                        # Stwórz znaczniki
                        ticks = []
                        current = first_tick
                        while current <= original_max:
                            ticks.append(current)
                            current += step
                        
                        # Dodaj skrajne wartości jeśli ich nie ma
                        if not ticks or ticks[0] > original_min:
                            ticks.insert(0, original_min)
                        if not ticks or ticks[-1] < original_max:
                            ticks.append(original_max)
                        
                        # Ogranicz liczbę znaczników do maksymalnie 7
                        if len(ticks) > 7:
                            # Zachowaj pierwsze, ostatnie i kilka środkowych
                            indices = [0]  # Pierwszy
                            middle_count = 5
                            for i in range(1, middle_count + 1):
                                idx = int(i * (len(ticks) - 1) / (middle_count + 1))
                                if idx not in indices:
                                    indices.append(idx)
                            indices.append(len(ticks) - 1)  # Ostatni
                            ticks = [ticks[i] for i in sorted(set(indices))]
                        
                        cbar.set_ticks(ticks)
                        # Formatuj etykiety - usuń zbędne miejsca dziesiętne dla liczb całkowitych
                        tick_labels = []
                        for tick in ticks:
                            if abs(tick - round(tick)) < 0.01:  # Praktycznie liczba całkowita
                                tick_labels.append(f'{int(round(tick))}')
                            else:
                                tick_labels.append(f'{tick:.1f}')
                        cbar.set_ticklabels(tick_labels)
                
                elif animation_type == 'flow':
                    # Mapa przepływów
                    data.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.7)
                    
                    # Dodaj przepływy
                    centroids = year_data.geometry.centroid
                    x_coords = centroids.x.values
                    y_coords = centroids.y.values
                    
                    # Napływ (dodatnie wartości)
                    positive_mask = values > 0
                    if np.any(positive_mask):
                        self._draw_convergent_flows(ax, x_coords[positive_mask], y_coords[positive_mask], 
                                                  values[positive_mask], max_val, 'blue', 'Inflow')
                    
                    # Odpływ (ujemne wartości)
                    negative_mask = values < 0
                    if np.any(negative_mask):
                        self._draw_divergent_flows(ax, x_coords[negative_mask], y_coords[negative_mask], 
                                                 np.abs(values[negative_mask]), max_val, 'red', 'Outflow')
                
                # Aktualizuj teksty
                year_text.set_text(f'Year: {year}')
                stats_text.set_text(f'Min: {np.min(values):.1f}\nMax: {np.max(values):.1f}\nMean: {np.mean(values):.1f}')
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.axis('off')
            
            return ax.collections + [year_text, stats_text]
        
        # Stwórz animację
        anim = FuncAnimation(fig, animate, frames=len(years), 
                           interval=1500, blit=False, repeat=True)
        
        # Zapisz animację
        try:
            anim.save(output_path, writer='pillow', fps=0.67)  # ~1.5 sekundy na klatkę
            print(f"Saved animation: {output_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            # Spróbuj zapisać jako mp4
            mp4_path = output_path.replace('.gif', '.mp4')
            try:
                anim.save(mp4_path, writer='ffmpeg', fps=0.67)
                print(f"Saved animation as MP4: {mp4_path}")
            except Exception as e2:
                print(f"Error saving animation as MP4: {e2}")
        
        plt.close()

    def _init_choropleth_animation(self, ax, data, min_val, max_val):
        """Inicjalizuje animację choropleth."""
        # Rysuj pustą mapę
        data.plot(ax=ax, color='lightgray', edgecolor='black')
        ax.set_title('')
        ax.axis('off')
        return None