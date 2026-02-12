import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import models
import torch.nn as nn

def get_densenet_model(num_classes=8):
    """Carga la arquitectura exacta que pide el archivo .pth"""
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    
    # Según el error, el modelo guardado tiene una capa intermedia de 512 
    # y termina en una capa '3'. Vamos a recrear esa estructura:
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),      # Esta es la capa que mide 512
        nn.ReLU(),                     # Capa 1
        nn.Dropout(0.2),               # Capa 2
        nn.Linear(512, num_classes)    # Esta es la capa '3' (final)
    )
    return model

def get_medical_cam(model, image_tensor, target_class_idx):
    """Grad-CAM Manual: Evita errores de Hook e Inplace en DenseNet."""
    model.eval()
    
    # 1. Extraemos las características manualmente hasta la última capa convolucional
    # image_tensor debe tener requires_grad=True
    with torch.enable_grad():
        features = model.features(image_tensor) # (1, 1024, 7, 7)
        features = features.detach().requires_grad_(True)
        
        # 2. Pasamos esas características por el clasificador
        # Necesitamos simular el Global Average Pooling y el Linear
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = model.classifier(out)
        
        score = out[:, target_class_idx]
        
        # 3. Calculamos gradientes de la puntuación respecto a las CARACTERÍSTICAS
        model.zero_grad()
        score.backward()
        
        grads = features.grad.detach() # Aquí están los gradientes
        activations = features.detach()

    # 4. Global Average Pooling de los gradientes (importancia de cada canal)
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    
    # 5. Mapa de calor (combinación lineal)
    cam = torch.sum(weights * activations, dim=1).squeeze().cpu().numpy()
    
    # ReLU y Normalización
    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    
    return cv2.resize(cam, (224, 224))

def show_medical_report(model, image, label_names, device='cpu'):
    model.to(device)
    model.eval()
    
    input_tensor = image.unsqueeze(0).to(device)
    
    # 1. Predicción rápida (sin gradientes para evitar líos)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)[0].cpu().numpy()
    
    top_idx = np.argmax(probs)
    
    # 2. Mapa de calor con el método manual (maneja sus propios gradientes)
    input_tensor.requires_grad = True
    heatmap = get_medical_cam(model, input_tensor, top_idx)
    
    # 3. Generamos el mapa de calor CON EL NUEVO MÉTODO
    heatmap = get_medical_cam(model, input_tensor, top_idx)
    
    # --- A partir de aquí, tu código de Matplotlib ---
    fig = plt.figure(figsize=(16, 10), facecolor='#f8f9fa')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])
    
    # Imagen Original
    ax0 = fig.add_subplot(gs[0, 0])
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    ax0.imshow(img_np, cmap='bone')
    ax0.set_title("ORIGINAL X-RAY", fontweight='bold')
    ax0.axis('off')

    # Mapa de Calor (AI FOCUS)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(img_np, cmap='bone')
    heatmap_colored = plt.cm.jet(heatmap)
    heatmap_colored[heatmap < 0.1] = 0 # Filtro suave de ruido
    ax1.imshow(heatmap_colored, alpha=0.4)
    ax1.set_title(f"AI FOCUS: {label_names[top_idx].upper()}", color='red', fontweight='bold')
    ax1.axis('off')

    # Gráfico de Barras
    ax2 = fig.add_subplot(gs[:, 1])
    y_pos = np.arange(len(label_names))
    colors = ['#d32f2f' if p > 0.5 else '#1976d2' for p in probs]
    bars = ax2.barh(y_pos, probs, color=colors, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(label_names, fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(x=0.5, color='#fbc02d', linestyle='--', label='Clinical Alarm')
    
    for bar in bars:
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.1%}', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig, probs