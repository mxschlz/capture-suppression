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
from scipy import signal
from SPACECUE_implicit import get_data_path

# --- Konfiguration ---
# ID des Probanden, für den die Abbildungen erstellt werden sollen.
SUBJECT_ID = '1'

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
picks = ['Fp1', 'Fz', 'Cz', 'Pz', 'Oz']
available_picks = [p for p in picks if p in raw.ch_names]

# Erzwinge das Matplotlib-Backend, um leere Plots beim Speichern zu vermeiden
mne.viz.set_browser_backend('matplotlib')

# Erstelle den Plot für einen 10-Sekunden-Ausschnitt.
# Wir beginnen bei 60 Sekunden, um einen Bereich mit wahrscheinlichen Artefakten zu finden.
fig = raw.plot(
	start=1123,
	duration=10,
	picks=available_picks,
	n_channels=len(available_picks),
	scalings=dict(eeg=150e-6),  # Skalierung anpassen, um Artefakte hervorzuheben (150 µV)
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

# Speichere die Abbildung in hoher Auflösung
fig_path = os.path.join(OUTPUT_PATH, 'slide_1_rohdaten_artefakte.png')
fig.savefig(fig_path, dpi=300)
print(f"Abbildung gespeichert: {fig_path}")
plt.close(fig)

# --- Slide 2: Das Prinzip (Cocktailparty-Effekt / BSS) ---

print("\nErstelle Abbildung für Slide 2: Schema der Blinden Quellentrennung mit echten Daten...")

# Pfade zu den benötigten abgeleiteten Dateien (laden wir direkt für Slide 2 & 3)
ica_path = os.path.join(DATA_PATH_BASE, f'derivatives/preprocessing/sci-{SUBJECT_ID}/eeg/sci-{SUBJECT_ID}_task-spacecue_ica.fif')
exclude_idx_path = os.path.join(DATA_PATH_BASE, f'derivatives/preprocessing/sci-{SUBJECT_ID}/eeg/sci-{SUBJECT_ID}_task-spacecue_ica_labels.txt')

# Lade die ICA-Lösung und die Artefakt-Indizes
ica = mne.preprocessing.read_ica(ica_path)
with open(exclude_idx_path, 'r') as f:
	exclude_idx = [int(line.strip()) for line in f.readlines()]

# Zeitfenster aus Slide 1 definieren (Ausschnitt mit den massiven Blinzlern)
tmin, tmax = 1128, 1133
start_idx, stop_idx = raw.time_as_index([tmin, tmax])
times = raw.times[start_idx:stop_idx]

# Kanäle auswählen (Fp1 für stärkstes Blinzeln, Pz als Referenz für echtes Gehirnsignal)
picks_eeg = ['Fp1', 'Cz']

# 1. Gemischte Signale (Roh-EEG abrufen)
data_raw = raw.get_data(picks=picks_eeg, start=start_idx, stop=stop_idx)

# 2. ICA Quellen isolieren
ica_sources = ica.get_sources(raw)
artifact_idx = exclude_idx[0] if exclude_idx else 0
brain_idx = [i for i in range(ica.n_components_) if i not in exclude_idx][0]
data_ica = ica_sources.get_data(picks=[artifact_idx, brain_idx], start=start_idx, stop=stop_idx)

# 3. Entmischte Signale (bereinigtes EEG erstellen)
raw_clean = raw.copy()
ica.apply(raw_clean, exclude=exclude_idx)
data_clean = raw_clean.get_data(picks=picks_eeg, start=start_idx, stop=stop_idx)

# Erstelle die Abbildung mit 3 Unter-Plots
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
fig.suptitle('Slide 2: Das Prinzip der Blinden Quellentrennung (mit echten EEG-Daten)', fontsize=16)

# Plot 1: Rohdaten
for i, ch_name in enumerate(picks_eeg):
	axes[0].plot(times, data_raw[i] * 1e6, label=f'{ch_name} (Gemischt)')
axes[0].set_title('1. Gemischte Signale (Gemessenes Roh-EEG an den Elektroden)')
axes[0].set_ylabel('Amplitude (µV)')
axes[0].legend(loc='upper right')

# Plot 2: ICA Komponenten
axes[1].plot(times, data_ica[0], label=f'ICA Komp. {artifact_idx} (Isoliertes Blinzel-Artefakt)', color='red')
axes[1].plot(times, data_ica[1], label=f'ICA Komp. {brain_idx} (z.B. Gehirn-Aktivität)', color='blue', alpha=0.7)
axes[1].set_title('2. Isolierte Ursprungssignale (Die "Instrumente")')
axes[1].set_ylabel('Aktivität (a.u.)')
axes[1].legend(loc='upper right')

# Plot 3: Bereinigtes EEG
for i, ch_name in enumerate(picks_eeg):
	axes[2].plot(times, data_clean[i] * 1e6, label=f'{ch_name} (Bereinigt)')
axes[2].set_title('3. Rekonstruierte Signale (EEG nach Stummschalten des Artefakts)')
axes[2].set_xlabel('Zeit (s)')
axes[2].set_ylabel('Amplitude (µV)')
axes[2].legend(loc='upper right')

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

if exclude_idx:
	# Plotte die Topographien der als Artefakte identifizierten Komponenten
	# Wir wählen bis zu 9 Komponenten für eine übersichtliche Darstellung
	picks_to_plot = exclude_idx[:min(len(exclude_idx), 9)]

	fig = ica.plot_components(
		picks=picks_to_plot,
		show=False
	)
	# MNE kann eine Liste von Figuren zurückgeben, wir nehmen die erste.
	if isinstance(fig, list):
		fig = fig[0]

	fig.suptitle(f'Proband {SUBJECT_ID}: Topographien von Artefakt-Komponenten', fontsize=16)

	# Speichere die Abbildung
	fig_path = os.path.join(OUTPUT_PATH, 'slide_3_ica_topographien.png')
	fig.savefig(fig_path, dpi=300)
	print(f"Abbildung gespeichert: {fig_path}")
	plt.close(fig)

else:
	print(f"Für Proband {SUBJECT_ID} wurden keine Komponenten zum Ausschluss markiert.")
	print("Plotte die ersten 9 Komponenten als Beispiel.")
	fig = ica.plot_components(picks=range(9), show=False, title=f'Proband {SUBJECT_ID}: Beispiel-Topographien')
	if isinstance(fig, list):
		fig = fig[0]
	fig_path = os.path.join(OUTPUT_PATH, 'slide_3_ica_topographien.png')
	fig.savefig(fig_path, dpi=300)
	print(f"Abbildung gespeichert: {fig_path}")
	plt.close(fig)
