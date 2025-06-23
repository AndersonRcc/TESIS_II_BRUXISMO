import os
import cv2
import torch
import numpy as np
from PIL import Image
from skimage import io, transform
from torchvision import models, transforms
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment.yolox import YOLOX
import torch.nn as nn
import warnings
import logging # <<-- Importa logging

warnings.filterwarnings("ignore", category=UserWarning)

# <<-- Configura el logger aquí para utils.py
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Nivel de log para este módulo

# Usar el dispositivo de forma global en utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"UTILS.PY: Dispositivo detectado globalmente: {device}")


# Cargar modelos (ejecutar solo una vez para ahorrar tiempo)
def cargar_modelos():
    logger.info("UTILS.PY: Iniciando la carga de modelos...")
    try:
        model_type = 'vit_b'
        checkpoint = './pretrained_model/tonguesam.pth'
        
        # Inicialización de YOLOX (asumiendo que YOLOX no carga modelos complejos en su init si no se le indica)
        logger.info("UTILS.PY: Inicializando YOLOX segmentador...")
        segment = YOLOX()
        logger.info("UTILS.PY: YOLOX segmentador inicializado.")

        logger.info(f"UTILS.PY: Cargando SAM model desde {checkpoint} a {device}...")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        sam_model.eval()
        logger.info("UTILS.PY: SAM model cargado y en modo evaluación.")

        logger.info("UTILS.PY: Cargando EfficientNetV2-S...")
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        
        model_state_dict_path = 'pretrained_model/best_model_ev2_tuned.pt'
        logger.info(f"UTILS.PY: Cargando state_dict para EfficientNetV2-S desde {model_state_dict_path} a {device}...")
        model.load_state_dict(torch.load(model_state_dict_path, map_location=device))
        model = model.to(device)
        model.eval()
        logger.info("UTILS.PY: EfficientNetV2-S cargado y en modo evaluación.")

        logger.info("UTILS.PY: Carga de todos los modelos completada exitosamente.")
        return sam_model, model, segment, device
    except Exception as e:
        logger.error(f"UTILS.PY: Error CRÍTICO durante la carga de modelos: {e}", exc_info=True)
        raise # Re-lanza la excepción para que app.py la capture


# Función para segmentar y predecir
def segmentar_y_predecir(image_path, sam_model, model, segment, device, etiquetas):
    logger.info(f"UTILS.PY: Iniciando segmentación y predicción para la imagen: {image_path}")
    try:
        with torch.no_grad():
            logger.info("UTILS.PY: Paso 1: Leyendo imagen con skimage.io.imread...")
            image_data = io.imread(image_path)
            logger.info(f"UTILS.PY: Paso 1: Imagen leída. Shape inicial: {image_data.shape}")

            if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
                image_data = image_data[:, :, :3]
                logger.info("UTILS.PY: Paso 1: Se eliminó el canal alfa de la imagen.")
            if len(image_data.shape) == 2:
                image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
                logger.info("UTILS.PY: Paso 1: Se convirtió la imagen a 3 canales (RGB).")

            logger.info("UTILS.PY: Paso 1: Aplicando normalización de percentiles y escalado a 255.")
            lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
            image_data_pre[image_data == 0] = 0
            logger.info(f"UTILS.PY: Paso 1: Escalando imagen a 400x400. Original shape: {image_data_pre.shape}")
            image_data_pre = transform.resize(image_data_pre, (400, 400), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
            image_data_pre = np.uint8(image_data_pre)
            logger.info(f"UTILS.PY: Paso 1: Pre-procesamiento inicial de imagen completado. Final shape: {image_data_pre.shape}")

            logger.info("UTILS.PY: Paso 2: Preparando imagen para SAM (ResizeLongestSide y preprocesamiento de SAM).")
            sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
            resize_img = sam_transform.apply_image(image_data_pre)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])
            logger.info(f"UTILS.PY: Paso 2: Input para image_encoder shape: {input_image.shape}")
            ts_img_embedding = sam_model.image_encoder(input_image)
            logger.info(f"UTILS.PY: Paso 2: Image embedding de SAM obtenido. Shape: {ts_img_embedding.shape}")

            img = image_data_pre # Reutilizar la imagen pre-procesada
            logger.info("UTILS.PY: Paso 3: Obteniendo bounding boxes con YOLOX.get_prompt...")
            boxes = segment.get_prompt(img)  # Esta función debe estar definida en YOLOX
            logger.info(f"UTILS.PY: Paso 3: Bounding boxes obtenidos de YOLOX: {boxes}")

            if boxes is not None and len(boxes) > 0: # Asegúrate de que boxes no sea None y no esté vacío
                logger.info("UTILS.PY: Paso 3: Transformando bounding box para SAM.")
                sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
                box = sam_trans.apply_boxes(boxes, (400,400)) # Tamaño de imagen original (400x400)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                logger.info(f"UTILS.PY: Paso 3: Bounding box transformado a tensor de PyTorch. Shape: {box_torch.shape}")
            else:
                box_torch = None
                logger.warning("UTILS.PY: Paso 3: No se detectaron bounding boxes. SAM podría no funcionar correctamente sin un prompt.")

            logger.info("UTILS.PY: Paso 4: Codificando prompts para SAM (sparse y dense embeddings)...")
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            logger.info("UTILS.PY: Paso 4: Prompts codificados.")

            logger.info("UTILS.PY: Paso 5: Decodificando máscara con SAM (mask_decoder)...")
            medsam_seg_prob, _ = sam_model.mask_decoder(
                image_embeddings=ts_img_embedding.to(device),
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            logger.info(f"UTILS.PY: Paso 5: Máscara SAM obtenida. Probabilidad shape: {medsam_seg_prob.shape}")

            logger.info("UTILS.PY: Paso 5: Post-procesando máscara SAM.")
            medsam_seg_prob = medsam_seg_prob.cpu().detach().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            medsam_seg = cv2.resize(medsam_seg, (400, 400)) # Re-escalar la máscara a 400x400
            logger.info(f"UTILS.PY: Paso 5: Máscara SAM final (binaria). Shape: {medsam_seg.shape}")

            logger.info("UTILS.PY: Paso 6: Aplicando máscara a la imagen original pre-procesada para obtener la región de interés.")
            # Asegúrate de que el tamaño de la máscara coincida con la imagen 'img' (que es 400x400)
            mask_resized = cv2.resize(medsam_seg, (img.shape[1], img.shape[0])) # Esto re-redimensiona a 400x400
            mask_bool = mask_resized > 0
            mask_expanded = np.expand_dims(mask_bool, axis=-1)
            tongue = img * mask_expanded
            logger.info("UTILS.PY: Paso 6: Región de interés (lengua) extraída.")

            output_pil = Image.fromarray(tongue)
            logger.info("UTILS.PY: Paso 6: Imagen segmentada convertida a PIL Image.")

            logger.info("UTILS.PY: Paso 7: Preparando imagen para el modelo de clasificación (EfficientNetV2-S).")
            val_test_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            input_tensor = val_test_transforms(output_pil).unsqueeze(0).to(device)
            logger.info(f"UTILS.PY: Paso 7: Input tensor para clasificación shape: {input_tensor.shape}")

            logger.info("UTILS.PY: Paso 8: Realizando inferencia con el modelo de clasificación.")
            model.eval() # Asegurarse de que el modelo esté en modo evaluación
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            logger.info(f"UTILS.PY: Paso 8: Inferencia de clasificación completada. Predicción raw index: {pred}.")

            probs_redondeadas = [round(p, 4) for p in probs.squeeze().tolist()]
            logger.info(f"UTILS.PY: Paso 8: Probabilidades redondeadas: {probs_redondeadas}.")

            logger.info("UTILS.PY: Segmentación y predicción finalizadas con éxito en utils.py.")
            return pred, probs_redondeadas, output_pil

    except Exception as e:
        logger.error(f"UTILS.PY: Error CRÍTICO en segmentar_y_predecir para {image_path}: {e}", exc_info=True)
        # Relanza la excepción para que app.py pueda manejarla y devolver un 500
        raise