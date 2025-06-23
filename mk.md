flowchart TD
    %% Cliente (Frontend)
    subgraph Cliente_Web [Cliente Web]
        UI[UI: HTML/CSS/JS]
        Captura[Captura / Carga de Imagen]
        Visual[Visualización del Resultado]
        PDF_User[Botón de Exportar PDF]
    end

    %% Servidor (Backend)
    subgraph Servidor_Backend [Servidor Backend]
        Recibir[Recepción de Imagen]
        Segmentar[TongueSAM - Segmentación]
        Preproc[Preprocesamiento<br><small>(OpenCV / NumPy)</small>]
        Clasificar[EfficientNetV2 - Clasificación]
        Resultado[Generación del Resultado]
        PDF[Generación del PDF<br><small>(pdfkit / reportlab)</small>]
    end

    %% Flujo del sistema
    Captura --> Recibir
    Recibir --> Segmentar
    Segmentar --> Preproc
    Preproc --> Clasificar
    Clasificar --> Resultado
    Resultado --> Visual

    PDF_User --> PDF
    PDF --> PDF_User

    %% Alineamiento
    UI --> Captura
    Visual --> PDF_User

    %% Estilos
    style Cliente_Web fill:#E3F2FD,stroke:#1E88E5,stroke-width:1px
    style Servidor_Backend fill:#FFF3E0,stroke:#FB8C00,stroke-width:1px
