import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import models
import torch.nn as nn
from typing import Tuple, Dict, List

# --- CONFIGURACIÓN DE ESTILO CORPORATIVO ---
MED_COLORS = {'primary': '#1a5276', 'alert': '#cb4335', 'neutral': '#ebedef', 'text': '#2c3e50'}
plt.rcParams['font.family'] = 'sans-serif'

def get_densenet_model(num_classes: int = 8) -> nn.Module:
    """
    Carga la arquitectura DenseNet121 optimizada para radiografía de tórax.
    Estructura adaptada para compatibilidad con pesos preentrenados específicos.
    """
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    
    # Arquitectura personalizada: Capa densa -> ReLU -> Dropout -> Salida
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model

def get_medical_cam(model: nn.Module, image_tensor: torch.Tensor, target_class_idx: int) -> np.ndarray:
    """
    Algoritmo Grad-CAM optimizado para DenseNet.
    Genera un mapa de activación de clase para transparencia diagnóstica.
    """
    model.eval()
    
    with torch.enable_grad():
        # Extracción de activaciones de la última capa convolucional
        features = model.features(image_tensor) 
        features = features.detach().requires_grad_(True)
        
        # Simulación de paso por clasificador
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = model.classifier(out)
        
        score = out[:, target_class_idx]
        model.zero_grad()
        score.backward()
        
        grads = features.grad.detach()
        activations = features.detach()

    # Ponderación de canales por importancia de gradiente
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze().cpu().numpy()
    
    # Procesamiento del Heatmap
    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    
    return cv2.resize(cam, (224, 224))

def show_medical_report(model: nn.Module, image: torch.Tensor, label_names: List[str], device: str = 'cpu') -> Tuple[plt.Figure, Dict]:
    """
    Genera un reporte visual de grado médico.
    Filtra hallazgos por relevancia clínica (>20% confianza).
    """
    model.to(device)
    model.eval()
    
    input_tensor = image.unsqueeze(0).to(device)
    
    # 1. Inferencia
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)[0].cpu().numpy()
    
    # 2. Filtrado de Hallazgos (Umbral de relevancia médica)
    THRESHOLD = 0.20
    results_dict = {label_names[i]: float(probs[i]) for i in range(len(label_names))}
    significant_findings = {k: v for k, v in results_dict.items() if v >= THRESHOLD}
    
    top_idx = np.argmax(probs)
    input_tensor.requires_grad = True
    heatmap = get_medical_cam(model, input_tensor, top_idx)
    
    # 3. Visualización Estilo DashBoard Médico
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1])
    
    # A. Imagen Original
    ax0 = fig.add_subplot(gs[0, 0])
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    ax0.imshow(img_np, cmap='bone')
    ax0.set_title("VISTA RADIOGRÁFICA ORIGINAL", fontsize=10, fontweight='bold', color=MED_COLORS['text'])
    ax0.axis('off')

    # B. AI Focus (Grad-CAM)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(img_np, cmap='bone')
    # Usamos mapa de color 'jet' pero con transparencia controlada
    heatmap_colored = plt.cm.jet(heatmap)
    heatmap_colored[heatmap < 0.15] = 0 
    ax1.imshow(heatmap_colored, alpha=0.35)
    ax1.set_title(f"ZONA DE INTERÉS: {label_names[top_idx].upper()}", fontsize=10, color=MED_COLORS['alert'], fontweight='bold')
    ax1.axis('off')

    # C. Análisis de Confianza Diagnóstica
    ax2 = fig.add_subplot(gs[:, 1])
    y_pos = np.arange(len(label_names))
    
    # Colores dinámicos: Rojo si supera el 50% (Crítico), Azul si es significativo, Gris el resto
    bar_colors = []
    for p in probs:
        if p > 0.5: bar_colors.append(MED_COLORS['alert'])
        elif p >= THRESHOLD: bar_colors.append(MED_COLORS['primary'])
        else: bar_colors.append(MED_COLORS['neutral'])
    
    bars = ax2.barh(y_pos, probs, color=bar_colors, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(label_names, fontsize=10, fontweight='bold', color=MED_COLORS['text'])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.0)
    
    # Línea de umbral clínico
    ax2.axvline(x=THRESHOLD, color=MED_COLORS['text'], linestyle='--', alpha=0.3, label='Clinical Threshold')
    
    # Etiquetas de porcentaje
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1%}', va='center', fontsize=9, fontweight='bold', 
                 color=MED_COLORS['text'] if width >= THRESHOLD else '#95a5a6')

    ax2.set_title("ÍNDICE DE CONFIANZA POR PATOLOGÍA", fontsize=12, fontweight='bold', pad=20)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig, significant_findings