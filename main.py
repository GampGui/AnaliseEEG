import mne
import matplotlib.pyplot as plt

# Lista padrão de 67 canais (10-20 expandido + extras)
channel_names_real = [
    "Fp1","Fpz","Fp2",
    "AF7","AF3","AFz","AF4","AF8",
    "F7","F5","F3","F1","Fz","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCz","FC2","FC4","FC6","FT8",
    "T7","C5","C3","C1","Cz","C2","C4","C6","T8",
    "TP7","CP5","CP3","CP1","CPz","CP2","CP4","CP6","TP8",
    "P7","P5","P3","P1","Pz","P2","P4","P6","P8",
    "PO7","PO3","POz","PO4","PO8",
    "O1","Oz","O2",
    "Iz",
    "FT9","FT10","TP9","TP10",
    "Extra1","Extra2","Extra3"  # canais auxiliares
]

def carregar_eeg(filepath):
    raw = mne.io.read_raw_fif(filepath, preload=True)

    # Se os canais forem genéricos (Ch1, Ch2...), renomeia automaticamente
    if all(ch.startswith("Ch") for ch in raw.ch_names):
        mapping = {f"Ch{i+1}": channel_names_real[i] 
                   for i in range(min(len(raw.ch_names), len(channel_names_real)))}
        raw.rename_channels(mapping)

    # Aplicar montagem padrão
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # Informações básicas
    print(raw.info)

    # Plotar sinais¢
    raw.plot(n_channels=20, scalings='auto')

    # Plotar mapa dos sensores
    try:
        raw.plot_sensors(show_names=True)
    except RuntimeError:
        print("Esse arquivo não tem posições válidas de canais para plot_sensors.")

    # Plotar espectro de potência
    raw.plot_psd(fmax=60)

    plt.show()
    return raw

# Exemplo de uso
raw = carregar_eeg(r"arquivos_dados\Doenca Parkinson Repouso\907_1_PD_REST1_raw.fif")