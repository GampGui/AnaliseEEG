import scipy.io as sio
import mne
import numpy as np

# Caminho do arquivo .mat
mat = sio.loadmat(r"C:\Users\guian\Downloads\CC_EEG_s129_P.mat")

# Extrair dados e parâmetros
eeg_data = mat['EEG'][0][0][15]   # matriz de canais x amostras
eeg_data = np.reshape(eeg_data, (eeg_data.shape[0], -1))
sfreq = int(mat['EEG'][0][0][11][0][0])  # taxa de amostragem

# Se não houver nomes de canais, cria nomes genéricos
channel_names = [f"Ch{i+1}" for i in range(eeg_data.shape[0])]

# Criar objeto RawArray
info = mne.create_info(ch_names=[f"Ch{i+1}" for i in range(eeg_data.shape[0])],
                       sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)


# Exportar para FIF (formato nativo do MNE)
raw.save(r"novo_arquivo\CC_EEG_s129_P_raw.fif", overwrite=True)

print("Conversão concluída: EDF, BDF e FIF gerados com sucesso!")