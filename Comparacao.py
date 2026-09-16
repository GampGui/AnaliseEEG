import mne
import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import ttk
import warnings

# Silenciar warnings do MNE
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")  # só mostra erros críticos

# Caminho da pasta com os dados de referência
pastaPato = "arquivos_dados"

# Percorre todas as subpastas e coleta arquivos .fif
arquivos_dados = []
for root, dirs, files in os.walk(pastaPato):
    for f in files:
        if f.endswith((".fif", ".fif.gz")):
            arquivos_dados.append(os.path.join(root, f))

# Se não encontrar nada, avisa
if not arquivos_dados:
    raise RuntimeError("Nenhum arquivo .fif encontrado em 'arquivos_dados'.")

# Extrai automaticamente o nome da subpasta como rótulo de patologia
patologias = [os.path.basename(os.path.dirname(f)) for f in arquivos_dados]

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
    "Extra1","Extra2","Extra3"
]

def calcular_assinatura(filepath):
    raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)

    # Renomeia canais se forem genéricos
    if all(ch.startswith("Ch") for ch in raw.ch_names):
        mapping = {f"Ch{i+1}": channel_names_real[i] 
                   for i in range(min(len(raw.ch_names), len(channel_names_real)))}
        raw.rename_channels(mapping)

    # Montagem padrão
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # Espectro de potência
    psd = raw.compute_psd(fmin=0.5, fmax=40, n_fft=1024)
    psd_data, freqs = psd.get_data(return_freqs=True)

    bandas = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 40)
    }

    assinatura = []
    for faixa, (fmin, fmax) in bandas.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        assinatura.append(psd_data[:, idx].mean())

    return np.array(assinatura)

def carregar_base(arquivos_dados, patologias):
    base = []
    for filepath, patologia in zip(arquivos_dados, patologias):
        assinatura = calcular_assinatura(filepath)
        base.append((assinatura, patologia, filepath))  # agora 3 elementos
    if not base:
        raise RuntimeError("Base de referência está vazia. Verifique os arquivos e rótulos.")
    return base


def mostrar_grafico_comparacao(assinatura_novo, assinatura_ref, tendencia, nome_arquivo):
    bandas = ["delta", "theta", "alpha", "beta", "gamma"]
    x = np.arange(len(bandas))
    width = 0.35

    plt.bar(x - width/2, assinatura_novo, width, label="Novo arquivo")
    plt.bar(x + width/2, assinatura_ref, width, label=f"Referência ({tendencia})")

    plt.xticks(x, bandas)
    plt.ylabel("Potência média")
    plt.title(f"Comparação espectral - {nome_arquivo}")
    plt.legend()
    plt.show()


def mostrar_psd(caminho_novo, caminho_ref, tendencia, nome_arquivo):
    print(f"Gerando PSD para {nome_arquivo} comparado com {tendencia}...")

    raw_novo = mne.io.read_raw_fif(caminho_novo, preload=True)
    raw_ref = mne.io.read_raw_fif(caminho_ref, preload=True)

   # Criar figura com 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSD do arquivo novo
    raw_novo.plot_psd(fmax=60, average=False, picks="eeg", ax=axes[0], show=False)
    axes[0].set_title(f"Arquivo em análise: {nome_arquivo}")

    # PSD da referência
    raw_ref.plot_psd(fmax=60, average=False, picks="eeg", ax=axes[1], show=False)
    axes[1].set_title(f"Referência: {tendencia}")

    plt.tight_layout()
    plt.show()


def criar_interface(novos_arquivos, base):
    root = tk.Tk()
    root.title("Comparação EEG")

    for arquivo in novos_arquivos:
        tendencia, assinatura_novo, assinatura_ref, caminho_ref = comparar_novo(arquivo, base)

        frame = ttk.Frame(root)
        frame.pack(pady=5, fill="x")

        label = ttk.Label(frame, text=f"{arquivo} → {tendencia}")
        label.pack(side="left", padx=10)

        botao1 = ttk.Button(frame, text="Gráfico de barras",
                           command=lambda a=assinatura_novo, r=assinatura_ref, t=tendencia, n=arquivo:
                           mostrar_grafico_comparacao(a, r, t, n))
        botao1.pack(side="right", padx=10)

        botao2 = ttk.Button(frame, text="Espectro de potência",
                            command=lambda cn=arquivo, cr=caminho_ref, t=tendencia, n=os.path.basename(arquivo):
                            mostrar_psd(cn, cr, t, n))
        botao2.pack(side="right", padx=10)

    root.mainloop()


def comparar_novo(novo_arquivo, base):
    assinatura_novo = calcular_assinatura(novo_arquivo)
    distancias = []
    for assinatura_ref, patologia, caminho_ref in base:
        distancia = np.linalg.norm(assinatura_novo - assinatura_ref)
        distancias.append((distancia, patologia, assinatura_ref, caminho_ref))

    if not distancias:
        return "Nenhuma referência disponível para comparação."

    # Ordena por distância
    distancias.sort(key=lambda x: x[0])
    distancia_min, tendencia, assinatura_ref, caminho_ref = distancias[0]

    # Exibição detalhada
    bandas = ["delta", "theta", "alpha", "beta", "gamma"]
    print(f"\nArquivo {os.path.basename(novo_arquivo)}")
    print(f"  ⇒ Tendência final: {tendencia} (distância {distancia_min:.4f})")
    print("  Espectro de potência médio por banda:")
    for nome, valor_novo, valor_ref in zip(bandas, assinatura_novo, assinatura_ref):
        print(f"    {nome:<6}: Arquivo em análise = {valor_novo:.4f} | Referência = {valor_ref:.4f}")

    return tendencia, assinatura_novo, assinatura_ref, caminho_ref








# Criar base
base = carregar_base(arquivos_dados, patologias)

# Pasta com novos arquivos
pasta_novo = "novo_arquivo"
novos_arquivos = [os.path.join(pasta_novo, f) for f in os.listdir(pasta_novo) if f.endswith(".fif")]


if not novos_arquivos:
    print("Nenhum arquivo novo encontrado em 'novo_arquivo'.")
else:

    criar_interface(novos_arquivos, base)

    for arquivo in novos_arquivos:
        resultado = comparar_novo(arquivo, base)
        print(f"Arquivo {os.path.basename(arquivo)} → Tendência à patologia: {resultado}")




