"""
Skript zur Erstellung von Abbildungen für einen EEG-Workshop zum Thema ICA.

Dieses Skript generiert drei Abbildungen, die die folgenden Konzepte illustrieren:
1.  Roh-EEG-Daten mit starken Artefakten (z.B. Blinzeln).
2.  Ein schematisches Diagramm des Prinzips der Blinden Quellentrennung (BSS).
3.  Topographien von ICA-Komponenten, die als Artefakte identifiziert wurden.

Das Skript nutzt die vorverarbeiteten Daten, die mit 'preproc_main.py' erstellt wurden.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mne
import numpy as np
import matplotlib.pyplot as plt
from SPACECUE import get_data_path
import seaborn as sns
import mne_icalabel

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

# Wende Average-Referenz an (sehr wichtig für eine korrekte räumliche Darstellung der Komponenten)
print("Wende Average-Referenz an...")
raw.set_eeg_reference(ref_channels="average")

# Zur Beschleunigung der ICA-Berechnung resampeln wir die Daten auf 250 Hz
print("Resample Daten auf 250 Hz zur Beschleunigung...")
raw = raw.resample(250)

# Wähle einige Kanäle aus, einschließlich frontaler Kanäle, auf denen Blinzeln gut sichtbar ist.
picks = ['Fp1', 'Cz', 'Pz', 'Oz', "TP10"]
available_picks = [p for p in picks if p in raw.ch_names]

# Erzwinge das Matplotlib-Backend, um leere Plots beim Speichern zu vermeiden
mne.viz.set_browser_backend('matplotlib')

# Erstelle den Plot für einen 10-Sekunden-Ausschnitt.
# Wir beginnen bei 4880 Sekunden, um einen Bereich mit deutlichen Artefakten zu finden.
fig = raw.plot(
	start=4880,
	duration=10,
	picks=available_picks,
	n_channels=len(available_picks),
	scalings=dict(eeg=150e-6),  # Skalierung auf 150 µV, damit sich große Blinzler weniger überlappen
	show=False,  # Plot nicht interaktiv anzeigen
	title=f'Proband {SUBJECT_ID}: Roh-EEG mit Augenartefakten',
	show_scrollbars=False  # UI-Scrollbars für eine saubere Präsentationsfolie ausblenden
)

# Mache Platz auf der rechten Seite des Plots, damit sich Linien und Inset nicht überlappen
fig.subplots_adjust(right=0.82)

# --- Topographie der ausgewählten Sensoren als Inset hinzufügen ---
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

print("\nErstelle Abbildung für Slide 2: Schema der Blinden Quellentrennung mit automatischer Klassifikation...")

print("Führe ICA auf den Rohdaten aus (dies kann einen Moment dauern)...")
# Für eine stabilere ICA-Berechnung filtern wir eine Kopie der Daten (1 Hz bis 100 Hz wie empfohlen)
raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=100.0, verbose=False)
ica = mne.preprocessing.ICA(n_components=15, method='infomax', fit_params=dict(extended=True), random_state=42, max_iter='auto')
ica.fit(raw_for_ica, verbose=False)

print("Klassifiziere ICA-Komponenten automatisch mit mne-icalabel...")
ic_labels = mne_icalabel.label_components(raw_for_ica, ica, method="iclabel")

components = ica.get_components() # shape: (n_channels, n_components)
ch_names = raw_for_ica.ch_names

# Finde Kanal-Indizes für die automatisierte Selektion
fp1_idx = ch_names.index("Fp1")
fp2_idx = ch_names.index("Fp2")
f7_idx = ch_names.index("F7")
f8_idx = ch_names.index("F8")
cz_idx = ch_names.index("Cz")
fcz_idx = ch_names.index("FCz")

# 1. Vertikales Auge / Blinzeln (Vertical Eye Blink)
vertical_eye_idx = None
eye_idxs = [i for i, l in enumerate(ic_labels['labels']) if l == 'eye blink']
if eye_idxs:
	blink_scores = []
	for idx in eye_idxs:
		w_fp1 = components[fp1_idx, idx]
		w_fp2 = components[fp2_idx, idx]
		if w_fp1 * w_fp2 > 0: # gleiches Vorzeichen (symmetrischer Blink)
			blink_scores.append((abs(w_fp1) + abs(w_fp2), idx))
	if blink_scores:
		vertical_eye_idx = max(blink_scores, key=lambda x: x[0])[1]

if vertical_eye_idx is None:
	vertical_eye_idx = max(range(ica.n_components_), key=lambda i: abs(components[fp1_idx, i]) + abs(components[fp2_idx, i]))

# 2. Horizontales Auge / Sakkaden (Horizontal Eye Saccade)
horizontal_eye_idx = None
saccade_scores = []
for idx in range(ica.n_components_):
	w_f7 = components[f7_idx, idx]
	w_f8 = components[f8_idx, idx]
	if w_f7 * w_f8 < 0: # entgegengesetztes Vorzeichen (laterale Augenbewegung)
		saccade_scores.append((abs(w_f7 - w_f8), idx))
if saccade_scores:
	saccade_scores.sort(reverse=True)
	for score, idx in saccade_scores:
		if idx != vertical_eye_idx:
			horizontal_eye_idx = idx
			break
if horizontal_eye_idx is None:
	horizontal_eye_idx = 2 if vertical_eye_idx != 2 else 0

# 3. Netzbrummen / Line Noise
line_noise_idx = None
line_noise_idxs = [i for i, l in enumerate(ic_labels['labels']) if l == 'line noise']
if line_noise_idxs:
	line_noise_idx = max(line_noise_idxs, key=lambda i: ic_labels['y_pred_proba'][i])
else:
	line_noise_idx = 3 if 3 not in [vertical_eye_idx, horizontal_eye_idx] else 4

# 4. Gehirnaktivität (Vertex / Auditory Brain)
brain_idxs = [i for i, l in enumerate(ic_labels['labels']) if l == 'brain']
vertex_brain_idx = None
if brain_idxs:
	vertex_brain_idx = max(brain_idxs, key=lambda i: abs(components[cz_idx, i]) + abs(components[fcz_idx, i]))

if vertex_brain_idx is None:
	non_artifacts = [i for i in range(ica.n_components_) if i not in [vertical_eye_idx, horizontal_eye_idx, line_noise_idx]]
	if non_artifacts:
		vertex_brain_idx = max(non_artifacts, key=lambda i: abs(components[cz_idx, i]) + abs(components[fcz_idx, i]))
	else:
		vertex_brain_idx = 1

print(f"  Vertikale Augen-Komponente (Blink): IC {vertical_eye_idx}")
print(f"  Horizontale Augen-Komponente (Saccade): IC {horizontal_eye_idx}")
print(f"  Netzbrummen-Komponente (Line Noise): IC {line_noise_idx}")
print(f"  Gehirnaktivität-Komponente (Vertex): IC {vertex_brain_idx}")

# Definiere auszuschließende Artefakte für die Rekonstruktion (alle außer brain und other)
exclude_idx = [idx for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])) if label not in ["brain", "other"]]
# Stelle sicher, dass die Augen- und Rausch-Komponenten ausgeschlossen werden
for idx in [vertical_eye_idx, horizontal_eye_idx, line_noise_idx]:
	if idx not in exclude_idx:
		exclude_idx.append(idx)
# Die Vertex-Komponente darf auf keinen Fall ausgeschlossen werden
if vertex_brain_idx in exclude_idx:
	exclude_idx.remove(vertex_brain_idx)

print(f"Auszuschließende Komponenten für die Bereinigung: {exclude_idx}")

# Zeitfenster aus Slide 1 definieren (Ausschnitt mit den massiven Blinzlern)
tmin, tmax = 4875, 4885
start_idx, stop_idx = raw.time_as_index([tmin, tmax])
times = raw.times[start_idx:stop_idx]

# Kanäle auswählen, die für die Artefakte und das Gehirnsignal repräsentativ sind
line_noise_max_ch = ch_names[np.argmax(np.abs(components[:, line_noise_idx]))]
picks_eeg = ['Fp1', 'F8', line_noise_max_ch, 'Cz']

# 1. Gemischte Signale (Roh-EEG abrufen)
data_raw = raw.get_data(picks=picks_eeg, start=start_idx, stop=stop_idx)

# 2. ICA Quellen isolieren
ica_sources = ica.get_sources(raw)
ica_picks = [vertical_eye_idx, horizontal_eye_idx, line_noise_idx, vertex_brain_idx]
data_ica = ica_sources.get_data(picks=ica_picks, start=start_idx, stop=stop_idx)

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

# Ein einheitlicher Offset für die Darstellung aller Plots
offset_y = 400.0  # 400 µV Abstand, wird nun auf alle angewendet

# --- Plot 1: Rohdaten ---
axes[0].set_title('1. Gemessenes Roh-EEG an den Elektroden')
for i, ch_data in enumerate(data_raw * 1e6):
	ch_data_centered = ch_data - np.mean(ch_data)
	y_pos = (len(picks_eeg) - 1 - i) * offset_y
	axes[0].plot(times, ch_data_centered + y_pos, color=color_raw, linewidth=1.5)

axes[0].set_yticks([(len(picks_eeg) - 1 - i) * offset_y for i in range(len(picks_eeg))])
axes[0].set_yticklabels(picks_eeg)
axes[0].set_ylabel('Amplitude (µV)')

# --- Skaliere ICA-Daten passend zu den EEG-Daten ---
# Damit die ICA-Komponenten im gleichen Maßstab wie das EEG dargestellt werden können:
data_raw_centered = np.array([ch - np.mean(ch) for ch in (data_raw * 1e6)])
data_ica_centered = np.array([ch - np.mean(ch) for ch in data_ica])

std_eeg = np.std(data_raw_centered)
std_ica = np.std(data_ica_centered)
ica_scaling_factor = std_eeg / std_ica if std_ica > 0 else 1.0

data_ica_scaled = data_ica_centered * ica_scaling_factor

# --- Plot 2: ICA Komponenten ---
axes[1].set_title('2. Isolierte Ursprungssignale (Die "Instrumente")')
yticks_ica = []
yticklabels_ica = []
ica_labels_mapping = {
	vertical_eye_idx: 'Vertikales Blinzeln',
	horizontal_eye_idx: 'Horizontale Augenbewegung',
	line_noise_idx: 'Netzbrummen',
	vertex_brain_idx: 'Auditive Aktivität'
}
for i, (idx, ch_data) in enumerate(zip(ica_picks, data_ica_scaled)):
	y_pos = (len(ica_picks) - 1 - i) * offset_y
	is_artifact = idx in exclude_idx
	label = ica_labels_mapping[idx]
	color = color_art if is_artifact else color_brain
	
	axes[1].plot(times, ch_data + y_pos, color=color, linewidth=1.5, label=label)
	yticks_ica.append(y_pos)
	yticklabels_ica.append(f'ICA {idx}')

axes[1].set_yticks(yticks_ica)
axes[1].set_yticklabels(yticklabels_ica)
axes[1].set_ylabel('Amplitude (µV)')


# --- Plot 3: Bereinigtes EEG ---
axes[2].set_title('3. Rekonstruierte Signale (Bereinigtes EEG)')
for i, ch_data in enumerate(data_clean * 1e6):
	ch_data_centered = ch_data - np.mean(ch_data)
	y_pos = (len(picks_eeg) - 1 - i) * offset_y
	axes[2].plot(times, ch_data_centered + y_pos, color=color_clean, linewidth=1.5)

axes[2].set_yticks([(len(picks_eeg) - 1 - i) * offset_y for i in range(len(picks_eeg))])
axes[2].set_yticklabels(picks_eeg)
axes[2].set_ylabel('Amplitude (µV)')
axes[2].set_xlabel('Zeit (s)')

# Finde die globalen y-Grenzen, um sicherzustellen, dass alle Plots auf derselben Skala sind
min_y, max_y = np.inf, -np.inf
for ax in axes:
	ax_min, ax_max = ax.get_ylim()
	min_y = min(min_y, ax_min)
	max_y = max(max_y, ax_max)

# Wende die globalen y-Grenzen auf alle Subplots an (mit etwas Puffer)
puffer = (max_y - min_y) * 0.05
for ax in axes:
	ax.set_ylim(min_y - puffer, max_y + puffer)

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

n_components_to_plot = 4
components_to_plot = [vertical_eye_idx, horizontal_eye_idx, line_noise_idx, vertex_brain_idx]
labels_topo = ['Vertikales Blinzeln', 'Horizontale Augenbewegung', 'Netzbrummen', 'Auditive Aktivität']

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
data_ica_complex = ica_sources.get_data(picks=components_to_plot, start=start_idx, stop=stop_idx)

offset_ica = 15.0  # Abstand weiter erhöht (auf 15.0), um Überlappungen bei den ICA-Komponenten zu verhindern
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, ch_data in enumerate(data_ica_complex):
	ch_data_centered = ch_data - np.mean(ch_data)
	y_pos = (len(components_to_plot) - 1 - i) * offset_ica
	ax2.plot(times, ch_data_centered + y_pos, color=colors[i], linewidth=1.8)

ax2.set_yticks([(len(components_to_plot) - 1 - i) * offset_ica for i in range(len(components_to_plot))])
ax2.set_yticklabels([f'ICA {c}\n({labels_topo[i]})' for i, c in enumerate(components_to_plot)])
ax2.set_title('2. Zeitverlauf von ICA-Komponenten')
ax2.set_xlabel('Zeit (s)')
ax2.set_xlim([times[0], times[-1]])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- Panel 3: Topographien ---
axes_topos = [fig.add_subplot(gs[2, i]) for i in range(n_components_to_plot)]
# MNE nimmt eine Liste von Axes entgegen und zeichnet die Topographien exakt dort ein
ica.plot_components(picks=components_to_plot, axes=axes_topos, show=False)

# Titel der Topographien einfärben, passend zu den Linien in Panel 2
for i, ax_topo in enumerate(axes_topos):
	ax_topo.set_title(f'ICA {components_to_plot[i]}\n({labels_topo[i]})', color=colors[i], fontweight='bold')

# Speichere die Abbildung
fig_path = os.path.join(OUTPUT_PATH, 'slide_3_ica_multipanel.png')
fig.savefig(fig_path, dpi=300)
print(f"Abbildung gespeichert: {fig_path}")
plt.close(fig)
