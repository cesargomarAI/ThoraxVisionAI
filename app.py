import streamlit as st
import torch
from PIL import Image
import numpy as np
from model_utils import get_densenet_model, show_medical_report
import torchvision.transforms as transforms

# Configuración de página con estilo MedTech
st.set_page_config(page_title="ThoraxVision AI", page_icon="🧬", layout="wide")

# CSS personalizado para ese aire de "Software Médico"
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1a237e; color: white; }
    .report-text { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_ai_model():
    # Cargamos la estructura y los pesos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_densenet_model()
    model.load_state_dict(torch.load('thoraxvision_final_v1.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2497/2497143.png", width=100) # Un icono médico
st.sidebar.title("ThoraxVision Control")
st.sidebar.markdown("---")
st.sidebar.info("Este sistema utiliza Deep Learning (DenseNet121) para asistir en el cribado radiológico.")


# --- CUERPO PRINCIPAL ---
st.title("🧬 ThoraxVision: Diagnóstico Asistido por IA")
st.write("Sube una radiografía de tórax en formato JPG o PNG para realizar el análisis.")

uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Imagen del Paciente', use_container_width=True)
    
    with col2:
        st.write("### ⚙️ Procesamiento de Datos")
        if st.button('Lanzar Análisis de Patologías'):
            with st.spinner('ThoraxVision está analizando los píxeles...'):
                model, device = load_ai_model()
                
                # Transformación de la imagen para el modelo
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                img_tensor = transform(image)
                label_names = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
                               "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
                
                # Generar el informe visual 
                fig, probs = show_medical_report(model, img_tensor, label_names, device)
                st.pyplot(fig, clear_figure=True)

                # --- SECCIÓN DE HALLAZGOS CLÍNICOS INTELIGENTES ---
                st.markdown("---")
                st.subheader("📝 Informe de Análisis Inteligente")

                # 1. Calibración Médica: Umbrales específicos por riesgo
                critical_pathologies = ['Pneumonia', 'Pneumothorax']
                critical_threshold = 0.25  # Más sensible para riesgos altos
                normal_threshold = 0.40

                findings = []
                for i, name in enumerate(label_names):
                    p = probs[i]
                    # Aplicamos el umbral según la patología
                    thresh = critical_threshold if name in critical_pathologies else normal_threshold
                    if p >= thresh:
                        findings.append({
                            'name': name, 
                            'prob': p, 
                            'is_critical': name in critical_pathologies
                        })

                # Ordenamos de mayor a menor probabilidad
                findings = sorted(findings, key=lambda x: x['prob'], reverse=True)

                if findings:
                    # Mostramos las métricas de forma dinámica
                    cols = st.columns(len(findings))
                    for idx, f in enumerate(findings):
                        with cols[idx]:
                            # Si es crítico o muy probable (>50%), marcamos como Urgente
                            is_urgent = f['is_critical'] or f['prob'] > 0.5
                            st.metric(
                                label=f['name'], 
                                value=f"{f['prob']:.1%}", 
                                delta="Prioridad Máxima" if is_urgent else "Revisar",
                                delta_color="inverse" if is_urgent else "normal"
                            )
                    
                    # 2. Mensaje inteligente basado en el hallazgo principal
                    main_f = findings[0]
                    st.warning(f"⚠️ **Hallazgo Principal:** El sistema detecta signos compatibles con **{main_f['name']}**.")
                    
                    # 3. Diagnóstico Diferencial (si hay más de una sospecha)
                    if len(findings) > 1:
                        others = [f['name'] for f in findings[1:]]
                        st.info(f"🔍 **Diagnóstico Diferencial:** Debido a la morfología observada, se sugiere descartar también: {', '.join(others)}.")
                    
                    st.write("👉 *Localización anatómica sugerida en el mapa de calor superior (AI FOCUS).*")
                else:
                    st.success("✅ **Estudio Normal:** No se observan hallazgos significativos por encima de los umbrales de seguridad.")
                
                st.success("Análisis completado con éxito.")
else:
    st.warning("Por favor, sube una imagen para activar el motor de inferencia.")

# Pie de página
st.markdown("---")
st.caption("Aviso: Esta herramienta es experimental y debe ser validada por un radiólogo colegiado.")