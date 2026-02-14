import streamlit as st
import torch
from PIL import Image
import numpy as np
from model_utils import get_densenet_model, show_medical_report
import torchvision.transforms as transforms

# 1. CONFIGURACIÓN DE PÁGINA PROFESIONAL
st.set_page_config(
    page_title="ThoraxVision AI | Diagnostic Support", 
    page_icon="🏥", 
    layout="wide"
)

# Estilo CSS para mejorar la UI médica
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .report-card { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #1a237e; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_ai_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_densenet_model()
    # Asegúrate de que el nombre del archivo sea el correcto
    model.load_state_dict(torch.load('thoraxvision_final_v1.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

# --- BARRA LATERAL (BRANDING) ---
with st.sidebar:
    try:
        logo = Image.open("logo.png")
        st.image(logo, use_container_width=True)
    except:
        st.title("🏥 ThoraxVision AI")
    
    st.divider()
    st.info("Sistema de soporte basado en Deep Learning (v1.2.4)")
    st.caption("Estandarización bajo protocolos de cribado radiológico.")

# --- CUERPO PRINCIPAL ---
st.title("🩺 ThoraxVision: Análisis Digital de Radiografía")
st.write("Carga de imágenes DICOM convertidas (JPG/PNG) para inspección por red neuronal.")

uploaded_file = st.file_uploader("Seleccionar radiografía de tórax (Vista Frontal)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col_img, col_proc = st.columns([1, 1])
    
    with col_img:
        st.image(image, caption='Fuente: Imagen del Paciente', use_container_width=True)
    
    with col_proc:
        st.markdown("### ⚙️ Centro de Inferencia")
        if st.button('EJECUTAR ANÁLISIS CLÍNICO'):
            with st.spinner('Procesando arquitectura DenseNet121...'):
                model, device = load_ai_model()
                
                # Transformaciones estándar de imagen médica
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                img_tensor = transform(image)
                label_names = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
                               "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
                
                # EJECUCIÓN DEL MODELO
                # fig: el gráfico de Matplotlib | results_dict: el diccionario {nombre: prob}
                fig, results_dict = show_medical_report(model, img_tensor, label_names, device)
                
                # Mostrar visualización técnica
                st.pyplot(fig, clear_figure=True)

                # --- INFORME CLÍNICO NIVEL ROCHE ---
                st.divider()
                st.markdown("### 📄 Informe de Hallazgos Digitales")
                st.caption(f"ID del Análisis: TV-{np.random.randint(1000, 9999)}")

                # Clasificación de Hallazgos
                # Cambiamos la lógica: ahora usamos results_dict (nombres) en vez de probs[i]
                high_conf = {k: v for k, v in results_dict.items() if v >= 0.45}
                mod_conf = {k: v for k, v in results_dict.items() if 0.20 <= v < 0.45}

                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("#### 🩺 Hallazgos Significativos")
                    if high_conf:
                        for name, prob in high_conf.items():
                            st.metric(label=name.upper(), value=f"{prob:.1%}", delta="SOSPECHA ALTA", delta_color="inverse")
                    else:
                        st.info("Sin anomalías críticas detectadas.")

                with c2:
                    st.markdown("#### 🔬 Diagnóstico Diferencial")
                    if mod_conf:
                        # Mostramos máximo 3 para no saturar al médico
                        for name, prob in list(mod_conf.items())[:3]:
                            st.warning(f"**{name}**: {prob:.1%}")
                        st.caption("Correlacionar con sintomatología clínica.")
                    else:
                        st.success("Estudio complementario normal.")

                # Nota de Interpretación
                with st.expander("👁️ Ver Interpretación Técnica de la IA"):
                    top_label = max(results_dict, key=results_dict.get)
                    st.write(f"""
                        **Impresión Radiológica:** La red detecta una morfología compatible con **{top_label}**. 
                        La zona de interés en el mapa de calor (AI FOCUS) indica la región con mayores activaciones convolucionales.
                    """)

                # Disclaimer Obligatorio MedTech
                st.markdown(
                    """
                    <div style="background-color: #fff4f4; padding: 15px; border-radius: 8px; border: 1px solid #ffcdd2;">
                        <p style="color: #b71c1c; font-size: 0.85rem; margin-bottom: 0;">
                            <b>⚠️ AVISO PARA PROFESIONALES DE LA SALUD:</b> Este análisis es una ayuda diagnóstica experimental. 
                            La decisión clínica final es responsabilidad exclusiva del médico radiólogo.
                        </p>
                    </div>
                    """, unsafe_allow_html=True
                )
else:
    st.info("Esperando carga de imagen para iniciar diagnóstico.")

# Footer
st.markdown("---")
st.caption("© 2026 ThoraxVision AI - MedTech Intelligence Division")