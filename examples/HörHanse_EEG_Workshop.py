"""
Skript zur Erstellung von Abbildungen für einen EEG-Workshop zum Thema ICA.

Dieses Skript generiert drei Abbildungen, die die folgenden Konzepte illustrieren:
1.  Roh-EEG-Daten mit starken Artefakten (z.B. Blinzeln).
2.  Ein schematisches Diagramm des Prinzips der Blinden Quellentrennung (BSS).
3.  Topographien von ICA-Komponenten, die als Artefakte identifiziert wurden.

Das Skript nutzt die vorverarbeiteten Daten, die mit 'preproc_main.py' erstellt wurden.
"""
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from SPACECUE import get_data_path
import seaborn as sns

sns.set_theme(context="talk", style="ticks")

# --- Konfiguration ---
# ID des Probanden, für den die Abbildungen erstellt werden sollen.
SUBJECT_ID = '2'

# Pfad, in dem die generierten Abbildungen gespeichert werden.
OUTPUT_PATH = './workshop_abbildungen'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Basis-Datenpfad aus dem Projekt abrufen
DATA_PATH_BASE = get_data_path()

# --- Slide 1: Warum ICA? (Problem: Artefakte in Rohdaten) ---

print("Erstelle Abbildung für Slide 1: Rohdaten mit Artefakten...")

# Pfad zur rohen EEG-Datei des Probanden
raw_path = os.path.join(DATA_PATH_BASE, f'sourcedata/raw/sci-{SUBJECT_ID}/eeg/sci-{SUBJECT_ID}_task-spacecue_raw.fif')

# Lade die Rohdaten
raw = mne.io.read_raw_fif(raw_path, preload=True)

# Referenzkanal 'Fz' hinzufügen, da dieser im Preprocessing ergänzt wurde.
# Ohne diesen Kanal verweigert die ICA die Anwendung, da die Kanalanzahl nicht übereinstimmt.
raw.add_reference_channels(['Fz'])

# Sensorpositionen (Montage) hinzufügen, da diese in den Rohdaten noch fehlen
montage = mne.channels.make_standard_montage("easycap-M1")
raw.set_montage(montage, on_missing='ignore')

# Wähle einige Kanäle aus, einschließlich frontaler Kanäle, auf denen Blinzeln gut sichtbar ist.
picks = ['Fp1', 'Cz', 'Pz', 'Oz', "TP10"]
available_picks = [p for p in picks if p in raw.ch_names]

# Erzwinge das Matplotlib-Backend, um leere Plots beim Speichern zu vermeiden
mne.viz.set_browser_backend('matplotlib')

# Erstelle den Plot für einen 10-Sekunden-Ausschnitt.
# Wir beginnen bei 60 Sekunden, um einen Bereich mit wahrscheinlichen Artefakten zu finden.
fig = raw.plot(
	start=4880,
	duration=10,
	picks=available_picks,
	n_channels=len(available_picks),
	scalings=dict(eeg=150e-6),  # Skalierung auf 250 µV erhöhen, damit sich große Blinzler weniger überlappen
	show=False,  # Plot nicht interaktiv anzeigen
	title=f'Proband {SUBJECT_ID}: Roh-EEG mit Augenartefakten',
	show_scrollbars=False  # UI-Scrollbars für eine saubere Präsentationsfolie ausblenden
)

# Mache Platz auf der rechten Seite des Plots, damit sich Linien und Inset nicht überlappen
fig.subplots_adjust(right=0.82)

# --- NEU: Topographie der ausgewählten Sensoren als Inset hinzufügen ---
# Achsen für das Inset erstellen (in dem neu geschaffenen Freiraum rechts)
ax_topo = fig.add_axes([0.83, 0.65, 0.15, 0.25])

# Kopie der Info nur für die gepickten Kanäle erstellen, um nur diese zu zeigen
info_picks = raw.copy().pick_channels(available_picks).info

# Topographie in das Inset plotten
mne.viz.plot_sensors(
	info_picks, 
	show_names=True, 
	axes=ax_topo, 
	show=False
)
# Hintergrund des Insets auf weiß setzen, damit eventuelle EEG-Linien nicht durchscheinen
ax_topo.set_facecolor('white')
ax_topo.set_title('Ausgewählte\nSensoren', fontsize=10)
sns.despine()

# Speichere die Abbildung in hoher Auflösung
fig_path = os.path.join(OUTPUT_PATH, 'slide_1_rohdaten_artefakte.png')
fig.savefig(fig_path, dpi=300)
print(f"Abbildung gespeichert: {fig_path}")
plt.close(fig)

# --- Slide 2: Das Prinzip (Cocktailparty-Effekt / BSS) ---

print("\nErstelle Abbildung für Slide 2: Schema der Blinden Quellentrennung mit echten Daten...")

print("Führe ICA auf den Rohdaten aus (dies kann einen Moment dauern)...")
# Für eine stabilere ICA-Berechnung filtern wir eine Kopie der Daten (1 Hz Highpass)
raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
ica = mne.preprocessing.ICA(n_components=15, random_state=42, max_iter='auto')
ica.fit(raw_for_ica, verbose=False)

print("\n" + "="*65)
print("INTERAKTIVE KOMPONENTEN-AUSWAHL")
print("="*65)
print("Die Topographien und Zeitverläufe der ICA-Komponenten werden nun angezeigt.")
print("1. Betrachte die Plots, um Blinzel-Artefakte und Gehirnaktivität zu identifizieren.")
print("2. BITTE SCHLIESSE DIE PLOT-FENSTER, wenn du fertig bist, um im Terminal deine Wahl einzugeben!")
print("="*65 + "\n")

# Topographien und Zeitverläufe interaktiv plotten
ica.plot_components(inst=raw, show=True)
ica.plot_sources(raw, show=True, block=True)

# Benutzereingabe abfragen
artifact_input = input("\nWelche ICA-Komponente(n) stellen Artefakte (z.B. Blinzeln) dar? (Komma-getrennt, z.B. 0, 1): ")
brain_input = input("Welche ICA-Komponente (einzelner Index) soll als 'Gehirnaktivität' in Abb. 2 gezeigt werden? (z.B. 2): ")

exclude_idx = [int(i.strip()) for i in artifact_input.split(',') if i.strip().isdigit()]
if not exclude_idx:
	print("Keine gültige Eingabe für Artefakte. Verwende Komponente 0 als Standard.")
	exclude_idx = [0]

if brain_input.strip().isdigit():
	brain_idx = int(brain_input.strip())
else:
	print("Keine gültige Eingabe für Gehirnaktivität. Verwende Komponente 1 als Standard.")
	brain_idx = 1 if 1 not in exclude_idx else 2

# Zeitfenster aus Slide 1 definieren (Ausschnitt mit den massiven Blinzlern)
tmin, tmax = 4875, 4885
start_idx, stop_idx = raw.time_as_index([tmin, tmax])
times = raw.times[start_idx:stop_idx]

# Kanäle auswählen (Fp1 für stärkstes Blinzeln, Pz als Referenz für echtes Gehirnsignal)
picks_eeg = ['Fp1', 'Cz']

# 1. Gemischte Signale (Roh-EEG abrufen)
data_raw = raw.get_data(picks=picks_eeg, start=start_idx, stop=stop_idx)

# 2. ICA Quellen isolieren
ica_sources = ica.get_sources(raw)
artifact_idx = exclude_idx[0]
data_ica = ica_sources.get_data(picks=[artifact_idx, brain_idx], start=start_idx, stop=stop_idx)

# 3. Entmischte Signale (bereinigtes EEG erstellen)
raw_clean = raw.copy()
ica.apply(raw_clean, exclude=exclude_idx)
data_clean = raw_clean.get_data(picks=picks_eeg, start=start_idx, stop=stop_idx)

# Erstelle die Abbildung mit 3 Unter-Plots
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

# Semantische Farbkodierung für den Workshop definieren
color_raw = 'gray'
color_art = 'crimson'
color_brain = 'royalblue'
color_clean = 'black'

# Feste Offsets für die Darstellung (um Überlappungen wie bei twinx zu vermeiden)
offset_eeg_uv = 400.0  # 400 µV Abstand für EEG-Kanäle
offset_ica = 8.0       # Abstand für ICA-Komponenten

# --- Plot 1: Rohdaten ---
axes[0].set_title('1. Gemessenes Roh-EEG an den Elektroden')
for i, ch_data in enumerate(data_raw * 1e6):
	ch_data_centered = ch_data - np.mean(ch_data)
	y_pos = (len(picks_eeg) - 1 - i) * offset_eeg_uv
	axes[0].plot(times, ch_data_centered + y_pos, color=color_raw, linewidth=1.5)

axes[0].set_yticks([(len(picks_eeg) - 1 - i) * offset_eeg_uv for i in range(len(picks_eeg))])
axes[0].set_yticklabels(picks_eeg)
axes[0].set_ylabel('Amplitude (µV)')

# --- Plot 2: ICA Komponenten ---
axes[1].set_title('2. Isolierte Ursprungssignale (Die "Instrumente")')
ica_art_centered = data_ica[0] - np.mean(data_ica[0])
axes[1].plot(times, ica_art_centered + offset_ica, color=color_art, linewidth=1.5, label='Artefakt (Blinzeln)')

ica_brain_centered = data_ica[1] - np.mean(data_ica[1])
axes[1].plot(times, ica_brain_centered + 0, color=color_brain, linewidth=1.5, label='Gehirnaktivität')

axes[1].set_yticks([offset_ica, 0])
axes[1].set_yticklabels([f'ICA {artifact_idx}', f'ICA {brain_idx}'])
axes[1].set_ylabel('Aktivität (a.u.)')
axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# --- Plot 3: Bereinigtes EEG ---
axes[2].set_title('3. Rekonstruierte Signale (Bereinigtes EEG)')
for i, ch_data in enumerate(data_clean * 1e6):
	ch_data_centered = ch_data - np.mean(ch_data)
	y_pos = (len(picks_eeg) - 1 - i) * offset_eeg_uv
	axes[2].plot(times, ch_data_centered + y_pos, color=color_clean, linewidth=1.5)

axes[2].set_yticks([(len(picks_eeg) - 1 - i) * offset_eeg_uv for i in range(len(picks_eeg))])
axes[2].set_yticklabels(picks_eeg)
axes[2].set_ylabel('Amplitude (µV)')
axes[2].set_xlabel('Zeit (s)')

sns.despine()
plt.tight_layout()

# Speichere die Abbildung
fig_path = os.path.join(OUTPUT_PATH, 'slide_2_bss_schema_echtedaten.png')
fig.savefig(fig_path, dpi=300)
print(f"Abbildung gespeichert: {fig_path}")
plt.close(fig)

# --- Slide 3: ICA-Komponenten erkennen ---

print("\nErstelle Abbildung für Slide 3: ICA-Komponenten-Topographien...")

# Die ICA-Komponente benötigt die Kanalinformationen aus den Rohdaten für den Plot.
# Da wir in Slide 1 bereits das 'raw' Objekt (inkl. Montage) geladen haben, können wir die Info direkt übernehmen!
ica.info = raw.info

print("Wir wählen die ersten 5 Hauptkomponenten (erklären die meiste Varianz) für die Übersicht.")
n_components_to_plot = 5
components = list(range(n_components_to_plot))

# Neues Figure-Layout mit GridSpec (3 Reihen)
fig = plt.figure(figsize=(14, 11), constrained_layout=True)
gs = fig.add_gridspec(3, n_components_to_plot, height_ratios=[1.5, 1.5, 1])

# --- Panel 1: Komplexe EEG Time-series ---
ax1 = fig.add_subplot(gs[0, :])
picks_eeg_complex = ['Fp1', 'Cz', 'Pz', 'Oz'] # Eine repräsentative Verteilung
picks_eeg_complex = [ch for ch in picks_eeg_complex if ch in raw.ch_names]
data_eeg_complex = raw.get_data(picks=picks_eeg_complex, start=start_idx, stop=stop_idx)

offset_eeg = 400e-6  # Auf 400 µV erhöht, damit massive Blinzler nicht überlappen
for i, ch_data in enumerate(data_eeg_complex):
	ch_data_centered = ch_data - np.mean(ch_data)
	y_pos = (len(picks_eeg_complex) - 1 - i) * offset_eeg
	ax1.plot(times, ch_data_centered + y_pos, color='darkslategray', linewidth=1.2)

ax1.set_yticks([(len(picks_eeg_complex) - 1 - i) * offset_eeg for i in range(len(picks_eeg_complex))])
ax1.set_yticklabels(picks_eeg_complex)
ax1.set_title('1. Gemessenes EEG')
ax1.set_xlim([times[0], times[-1]])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Panel 2: ICA Time-series ---
ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
data_ica_complex = ica_sources.get_data(picks=components, start=start_idx, stop=stop_idx)

offset_ica = 15.0  # Abstand weiter erhöht (auf 15.0), um Überlappungen bei den ICA-Komponenten zu verhindern
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, ch_data in enumerate(data_ica_complex):
	ch_data_centered = ch_data - np.mean(ch_data)
	y_pos = (len(components) - 1 - i) * offset_ica
	ax2.plot(times, ch_data_centered + y_pos, color=colors[i], linewidth=1.8)

ax2.set_yticks([(len(components) - 1 - i) * offset_ica for i in range(len(components))])
ax2.set_yticklabels([f'ICA {c}' for c in components])
ax2.set_title('2. Zeitverlauf von ICA-Komponenten')
ax2.set_xlabel('Zeit (s)')
ax2.set_xlim([times[0], times[-1]])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- Panel 3: Topographien ---
axes_topos = [fig.add_subplot(gs[2, i]) for i in range(n_components_to_plot)]
# MNE nimmt eine Liste von Axes entgegen und zeichnet die Topographien exakt dort ein
ica.plot_components(picks=components, axes=axes_topos, show=False)

# Titel der Topographien einfärben, passend zu den Linien in Panel 2
for i, ax_topo in enumerate(axes_topos):
	ax_topo.set_title(f'ICA {components[i]}', color=colors[i], fontweight='bold')

# Speichere die Abbildung
fig_path = os.path.join(OUTPUT_PATH, 'slide_3_ica_multipanel.png')
fig.savefig(fig_path, dpi=300)
print(f"Abbildung gespeichert: {fig_path}")
plt.close(fig)
