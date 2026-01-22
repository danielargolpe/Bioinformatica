import pandas as pd
import warnings
import re
import matplotlib.pyplot as plt
import numpy as np

# ----- BLOQUE I: Parsea, limpia y levanta un VCF anotado -----
# --- BLOQUE 1A: Parsea un strig ---
def parse_info_field(info_string):
    """
    Parsea un string de INFO (ej. "DP=10;AF=0.5;DB")
    en un diccionario, manejando 'flags' (ej. "DB").
    """
    info_dict = {}
    
    # Manejar caso de que INFO esté vacío ('.') o sea un float (NaN)
    if not isinstance(info_string, str) or info_string == '.':
        return pd.Series(info_dict)

    fields = info_string.split(';')
    for field in fields:
        parts = field.split('=')
        if len(parts) == 2:
            # Es un par 'KEY=VALUE'
            info_dict[parts[0]] = parts[1]
        elif len(parts) == 1 and parts[0] != '':
            # Es un 'FLAG'
            info_dict[parts[0]] = True # Asignamos True para indicar presencia
            
    return pd.Series(info_dict)

#--- BLOQUE 1B: Limpia y levanta un VCF anotado ---
def limpiar_y_cargar_vcf(vcf_path):
    """
    Carga un archivo VCF en un DataFrame de Pandas,
    limpiando el header y parseando las columnas INFO y FORMAT.
    """
    print(f"Cargando VCF desde: {vcf_path}")
    
    # 1. Contar líneas de metadatos (##)
    header_lines_to_skip = 0
    try:
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('##'):
                    header_lines_to_skip += 1
                else:
                    break
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {vcf_path}")
        return None
    
    print(f"Saltando {header_lines_to_skip} líneas de metadatos '##'...")

    # 2. Leer el VCF
    try:
        df = pd.read_csv(vcf_path, sep='\t', skiprows=header_lines_to_skip)
    except pd.errors.EmptyDataError:
        print("Error: El VCF no contiene variantes (solo header).")
        return None

    # 3. Liampiar nombre de columna
    df.rename(columns={'#CHROM': 'CHROM'}, inplace=True)

    print("VCF cargado. Parseando columnas INFO y FORMAT...")

    # 4. Parsear la columna 'INFO'
    info_df = df['INFO'].apply(parse_info_field)
    
    # Convertir a números lo que se pueda (ignorar 'True'/errores)
    info_df = info_df.apply(pd.to_numeric, errors='ignore')

    # 5. Parsear la columna 'FORMAT' y la muestra
    sample_col = df.columns[-1]
    
    # Manejar VCFs sin muestras
    if 'FORMAT' not in df.columns:
        print("VCF no contiene columnas FORMAT/SAMPLE. Devolviendo solo INFO.")
        main_df = df.drop(columns=['INFO'])
        info_df.columns = pd.MultiIndex.from_product([['INFO'], info_df.columns])
        final_df = pd.concat([main_df, info_df], axis=1)
        return final_df

    # Asumimos que el formato es el mismo (tomamos el primero no nulo)
    first_valid_format = df['FORMAT'].dropna().iloc[0]
    format_keys = first_valid_format.split(':')
    
    # Función para parsear la muestra, manejando valores nulos ('.')
    def parse_sample_field(sample_string):
        if not isinstance(sample_string, str) or sample_string == '.':
             return pd.Series([None] * len(format_keys), index=format_keys)
        return pd.Series(sample_string.split(':'), index=format_keys)

    sample_df = df[sample_col].apply(parse_sample_field)

    # 6. Crear MULTI-ÍNDICE
    info_df.columns = pd.MultiIndex.from_product([['INFO'], info_df.columns])
    sample_df.columns = pd.MultiIndex.from_product([[sample_col], sample_df.columns])

    # 7. Combinar todo en el Dataframe final 
    main_df = df.drop(columns=['INFO', 'FORMAT', sample_col])
    final_df = pd.concat([main_df, info_df, sample_df], axis=1)

    return final_df

    # --- BLOQUE II: Análisis de Tipos de SNV ---
    
def analizar_distribucion_snv(vcf_df):
    """
    Analiza un DataFrame de variantes VCF, filtra los SNVs y calcula
    el ratio Transición/Transversión (Ts/Tv).
    Args: vcf_df (pd.DataFrame): DataFrame con columnas 'REF' y 'ALT'.
    Returns: float: El ratio Ts/Tv calculado (o None si no se pudo calcular).
    """
    print("\n--- Bloque 2A: Análisis de Tipos de SNV (REF > ALT) ---")

    # 1. Definir bases válidas
    valid_bases = ['A', 'C', 'G', 'T']
    
    # Validar que el DF no sea None
    if vcf_df is None or vcf_df.empty:
        print("Error: El DataFrame está vacío o es None.")
        return None

    # 2. Filtrar para obtener solo SNVs
    #    Nota: Usamos vcf_df (el argumento) en lugar de la variable global
    snv_df = vcf_df[
        (vcf_df['REF'].str.len() == 1) &
        (vcf_df['ALT'].str.len() == 1) &
        (vcf_df['REF'].isin(valid_bases)) &
        (vcf_df['ALT'].isin(valid_bases))
    ].copy()

    if snv_df.empty:
        print("No se encontraron SNVs (variantes de un solo nucleótido) en el VCF.")
        return None
    else:
        print(f"\nTotal de variantes en el VCF: {len(vcf_df)}")
        print(f"Número de SNVs puros encontrados: {len(snv_df)}")

        # 3. Crear la columna de tipo de variante
        snv_df['variant_type'] = snv_df['REF'] + '>' + snv_df['ALT']

        # 4. Calcular el NÚMERO (conteo)
        print("\n--- Conteo de Tipos de SNV ---")
        snv_counts = snv_df['variant_type'].value_counts().sort_index()
        print(snv_counts)

        # 5. Calcular la FRACCIÓN (porcentaje)
        print("\n--- Fracción de Tipos de SNV ---")
        snv_fractions = snv_df['variant_type'].value_counts(normalize=True).sort_index()
        print((snv_fractions * 100).round(2).astype(str) + ' %')

        # --- Análisis de Resultados (Transiciones vs Transversiones) ---
        print("\n--- Análisis del Resultado: Ratio Transición/Transversión (Ts/Tv) ---")
        
        transitions = ['A>G', 'G>A', 'C>T', 'T>C']
        
        snv_df['ts_tv_type'] = snv_df['variant_type'].apply(
            lambda x: 'Transition' if x in transitions else 'Transversion'
        )
        
        ts_tv_counts = snv_df['ts_tv_type'].value_counts()
        
        print("Conteo de Transiciones vs. Transversiones:")
        print(ts_tv_counts)
        
        # Calcular el ratio Ts/Tv
        ts_tv_ratio = None
        try:
            ts_count = ts_tv_counts.get('Transition', 0)
            tv_count = ts_tv_counts.get('Transversion', 0)
            
            if tv_count == 0:
                print("\nRatio Ts/Tv: Infinito (No se encontraron transversiones)")
            else:
                ts_tv_ratio = ts_count / tv_count
                print(f"\n** Ratio Ts/Tv (Transiciones / Transversiones): {ts_tv_ratio:.3f} **")
                
                print("\n**Interpretación del Ratio Ts/Tv:**")
                print(" * Para **exomas humanos**, se espera un ratio de ~3.0 (aprox. 2.8-3.3).")
                print(" * Para **genomas completos humanos**, se espera un ratio de ~2.1.")
                print(" * Un ratio muy bajo (ej. < 1.5) suele indicar un alto número de Falsos Positivos.")

        except Exception as e:
            print(f"No se pudo calcular el ratio Ts/Tv: {e}")
            
        return ts_tv_ratio

# BLOQUE 2 B) Análisis de Anotaciones de Variantes
def analizar_consecuencias_biologicas(vcf_df, annotation_column=('INFO', 'ANN')):
    """
    Analiza las anotaciones funcionales (SnpEff/VEP) del VCF.
    Calcula fracciones de Exónica/Intrónica y Missense/Nonsense/Silent.
    Args: vcf_df (pd.DataFrame): DataFrame con los datos.
    annotation_column (tuple/str): Nombre de la columna con la anotación (ej. ('INFO', 'ANN')).
    Returns:pd.DataFrame: El DataFrame original con nuevas columnas de categorías añadidas.
    """
    print("\n--- Módulo (b): Análisis de Anotaciones de Variantes ---")

    # 1. Validación de columna
    if annotation_column not in vcf_df.columns:
        print(f"Error: No se encontró la columna de anotación {annotation_column}.")
        print("Verifica si es ('INFO', 'ANN') (SnpEff) o ('INFO', 'CSQ') (VEP).")
        return vcf_df # Devolvemos el DF sin cambios

    # --- Helper interno para parsear el string de SnpEff ---
    def get_primary_consequence(ann_field):
        if not isinstance(ann_field, str):
            return 'No_Annotation'
        try:
            # SnpEff format: Alelo|Anotación|Impacto|Gene...
            first_annotation = ann_field.split(',')[0]
            consequence = first_annotation.split('|')[1]
            return consequence.split('&')[0]
        except (IndexError, AttributeError):
            return 'Parse_Error'

    # 2. Extraer la Consecuencia
    # Usamos .copy() para no afectar advertencias de 'SettingWithCopy'
    vcf_df = vcf_df.copy()
    vcf_df['Consequence'] = vcf_df[annotation_column].apply(get_primary_consequence)
    
    # 3. Categorización General (Exón/Intrón/Otro)
    print("\n--- Nivel 1: Categorías Generales ---")
    
    exonic_terms = [
        'missense_variant', 'synonymous_variant', 'stop_gained', 
        'stop_lost', 'start_lost', 'stop_retained_variant',
        '3_prime_UTR_variant', '5_prime_UTR_variant', 'coding_sequence_variant',
        'frameshift_variant'
    ]
    intronic_terms = ['intron_variant']
    
    def categorize_general(consequence):
        if consequence in exonic_terms:
            return 'Exónica'
        elif consequence in intronic_terms:
            return 'Intrónica'
        else:
            return 'Otro'

    vcf_df['General_Category'] = vcf_df['Consequence'].apply(categorize_general)

    # Imprimir resultados Nivel 1
    general_counts = vcf_df['General_Category'].value_counts()
    general_fractions = vcf_df['General_Category'].value_counts(normalize=True)
    print((general_fractions * 100).round(2).astype(str) + ' %')

    # 4. Subcategorización Exónica (Missense/Nonsense/Silent)
    print("\n--- Nivel 2: Subcategorías Exónicas ---")
    
    # Filtramos solo para calcular estadísticas, pero las etiquetas van al DF principal
    exonic_mask = vcf_df['General_Category'] == 'Exónica'
    
    if not exonic_mask.any():
        print("No se encontraron variantes exónicas.")
    else:
        exonic_sub_map = {
            'missense_variant': 'Missense',
            'stop_gained': 'Nonsense',
            'synonymous_variant': 'Silent'
        }
        
        # Aplicamos el mapeo solo a las filas exónicas
        vcf_df.loc[exonic_mask, 'Exonic_Subcategory'] = vcf_df.loc[exonic_mask, 'Consequence'].map(exonic_sub_map)
        # Llenamos NaN (las exónicas que no son miss/non/silent) con 'Otro_Exonico'
        vcf_df.loc[exonic_mask, 'Exonic_Subcategory'] = vcf_df.loc[exonic_mask, 'Exonic_Subcategory'].fillna('Otro_Exonico')
        
        # Calcular estadísticas solo del subset exónico
        exonic_stats = vcf_df.loc[exonic_mask, 'Exonic_Subcategory'].value_counts(normalize=True)
        print((exonic_stats * 100).round(2).astype(str) + ' %')

    return vcf_df

# BLOQUE 2: C) Análisis de Sustituciones Missense
def analizar_sustituciones_missense(vcf_df, annotation_column=('INFO', 'ANN')):
    """
    Analiza las variantes Missense para calcular la frecuencia de cada 
    sustitución de aminoácidos (ej. Ala>Gly).
    Args: vcf_df (pd.DataFrame): DataFrame que ya pasó por el Módulo (b) 
            y contiene las columnas 'Exonic_Subcategory' y 'Consequence'.
        annotation_column (tuple/str): Nombre de la columna original de SnpEff.
    Returns: None: Solo imprime los resultados del análisis.
    """
    print("\n--- Bloque 2 C): Análisis de Sustituciones Missense (Aminoácidos) ---")

    # 1. Validación de columna necesaria
    if 'Exonic_Subcategory' not in vcf_df.columns:
        print("Error: La columna 'Exonic_Subcategory' no existe. Ejecuta el Bloque 2B) primero.")
        return

    # 2. Filtrar solo las variantes Missense
    missense_df = vcf_df[
        vcf_df['Exonic_Subcategory'] == 'Missense'
    ].copy()

    if missense_df.empty:
        print("No se encontraron variantes 'Missense' para analizar.")
        return
    else:
        print(f"Analizando {len(missense_df)} variantes 'Missense'...")
        
        # 3. Definir los 20 aminoácidos 
        aa_codes_3_letter = {
            "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", 
            "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", 
            "Trp", "Tyr", "Val"
        }

        def get_aa_substitution(ann_field):
            """
            Parsea el campo ANN de SnpEff para encontrar la notación p. (HGVSp).
            Formato esperado: p.Ala123Gly
            """
            if not isinstance(ann_field, str):
                return None
            try:
                # 1. Tomar la primera anotación
                first_ann = ann_field.split(',')[0]
                # 2. Tomar el campo HGVSp (p.) (índice 10 en SnpEff)
                p_notation = first_ann.split('|')[10]

                # 3. Parsear el HGVSp (ej. "p.Ala123Gly")
                if p_notation.startswith('p.') and len(p_notation) > 3:
                    change_str = p_notation[2:]
                    
                    # Regex: (3 letras)(números)(3 letras)
                    match = re.match(r'([A-Za-z]{3})(\d+)([A-Za-z]{3})', change_str)
                    
                    if match:
                        ref_aa = match.group(1).capitalize()
                        alt_aa = match.group(3).capitalize()
                        
                        # 4. Validar que son AA y no son iguales
                        if ref_aa in aa_codes_3_letter and \
                           alt_aa in aa_codes_3_letter and \
                           ref_aa != alt_aa:
                            
                            return f"{ref_aa}>{alt_aa}"
                
                return None
            
            except (IndexError, AttributeError):
                return None

        # 4. Aplicar la función de parseo
        # ¡IMPORTANTE! Aquí usamos el DataFrame filtrado (missense_df) para aplicar el parseo 
        # a la columna original de anotación (annotation_column)
        missense_df['AA_Change'] = missense_df[annotation_column].apply(get_aa_substitution)

        # 5. Filtrar los que no se pudieron parsear (None)
        valid_missense = missense_df.dropna(subset=['AA_Change'])
        
        if valid_missense.empty:
            print("Se encontraron variantes missense, pero no se pudo parsear el cambio de AA (formato HGVSp no encontrado o no válido).")
        else:
            # 6. Calcular NÚMERO
            print("\n--- Conteo de Sustituciones de Aminoácidos (Top 20) ---")
            aa_counts = valid_missense['AA_Change'].value_counts()
            print(aa_counts.head(20))

            # 7. Calcular FRACCIÓN
            print("\n--- Fracción de Sustituciones de Aminoácidos (Top 20) ---")
            aa_fractions = valid_missense['AA_Change'].value_counts(normalize=True)
            print((aa_fractions * 100).round(2).astype(str).head(20))
            
            total_unique_changes = len(aa_counts)
            print(f"\n... (Total de {total_unique_changes} tipos de sustitución únicos encontrados) ...")

# --- BLOQUE II: D) Merge VCF con gnomAD ---

def agregar_anotaciones_gnomad(vcf_df):
    """
    Carga un CSV de gnomAD, harmoniza los prefijos de cromosoma y fusiona 
    (merge) los datos con el DataFrame VCF existente basándose en una clave única.
    Args: vcf_df (pd.DataFrame): DataFrame VCF principal (mi_vcf_df).
    Returns: pd.DataFrame: El DataFrame original con las columnas de gnomAD añadidas.
    """
    print("\n--- Bloque 2 D): Agregando anotaciones de gnomAD ---")

    # --- 1. CONFIGURACIÓN ---
    gnomad_csv_file = "/mnt/c/Backup_compu/INIDEP/CURSOS/Bioinformatica_avanzada/TP7/gnomAD_v2.1.1_ENST00000335295_2023_09_22_15_18_19.csv"
    
    gnomad_cols_to_add = [
        'ClinVar Clinical Significance', 
        'Allele Frequency', 
        'Homozygote Count'
    ]
    
    # Nombres de las columnas clave en el CSV de gnomAD 
    gnomad_key_cols = {
        'chrom': 'Chromosome',  
        'pos': 'Position',      
        'ref': 'Reference',     
        'alt': 'Alternate'      
    }

    # --- 2. Cargar el CSV de gnomAD ---
    try:
        print(f"Cargando gnomAD CSV desde: {gnomad_csv_file}")
        
        # Leemos el CSV.
        gnomad_df = pd.read_csv(
            gnomad_csv_file, 
            sep=',', # Separador de comas (CSV)
            low_memory=False
        )
        
        # 3. CORRECCIÓN DE TIPO: Limpiar y convertir la posición de gnomAD
        # Usamos 'Int64' (con I mayúscula) para manejar valores NaN en columnas enteras.
        # Si esta línea falla, el 'except' principal lo capturará.
        gnomad_df[gnomad_key_cols['pos']] = gnomad_df[gnomad_key_cols['pos']].astype('Int64')
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo gnomAD en {gnomad_csv_file}. Saltando merge.")
        return vcf_df
        
    except Exception as e:
        # Este 'except' captura cualquier error en la carga o en la conversión 'astype('Int64')'
        print(f"Error al cargar/procesar el CSV de gnomAD: {e}. Merge saltado.")
        # Opcional: print("Verifique si la columna de POSICIÓN contiene texto o valores NO numéricos.")
        return vcf_df
    
    # --- 4. ARMONIZACIÓN DE PREFIJOS (FIX) ---
    print("Armonizando prefijos de cromosomas (chr -> '')...")
    
    # Quitar 'chr' del DataFrame gnomAD
    gnomad_chrom_col = gnomad_key_cols['chrom'] 
    gnomad_df[gnomad_chrom_col] = gnomad_df[gnomad_chrom_col].astype(str).str.replace('chr', '')
    
    # Quitar 'chr' del DataFrame VCF
    vcf_df['CHROM'] = vcf_df['CHROM'].astype(str).str.replace('chr', '')

    # --- 5. Crear la Columna Llave ("merge key") ---
    print("Creando llaves únicas de variante para el merge...")
    
    # 5a. Crear llave en el DataFrame VCF
    vcf_df['merge_key'] = vcf_df['CHROM'].astype(str) + '-' + \
                          vcf_df['POS'].astype(str) + '-' + \
                          vcf_df['REF'].astype(str) + '-' + \
                          vcf_df['ALT'].astype(str)

    # 5b. Crear llave en el DataFrame gnomAD
    gnomad_df['merge_key'] = gnomad_df[gnomad_key_cols['chrom']].astype(str) + '-' + \
                             gnomad_df[gnomad_key_cols['pos']].astype(str) + '-' + \
                             gnomad_df[gnomad_key_cols['ref']].astype(str) + '-' + \
                             gnomad_df[gnomad_key_cols['alt']].astype(str)
                             
    # --- 6. Realizar el Merge ---
    print("Realizando el 'merge' (left join) entre el VCF y gnomAD...")
    
    # Seleccionamos solo la llave y las columnas target
    gnomad_subset_df = gnomad_df[['merge_key'] + gnomad_cols_to_add]
    
    merged_df = pd.merge(
        vcf_df,
        gnomad_subset_df,
        on='merge_key',
        how='left'
    )
    
    # Limpiamos la columna llave
    merged_df.drop(columns=['merge_key'], inplace=True)
    print(" Merge completado.")
    
    return merged_df


# Bloque II E) Comparación gráfica de la frecuencia alélica del VCF (AF) con la de gnomAD
def comparar_frecuencias_gnomad(df):
    """
    Compara gráficamente la frecuencia alélica (AF) de tu VCF con la frecuencia 
    poblacional de gnomAD, usando las columnas correctas del INFO.
    
    Args:
        df (pd.DataFrame): DataFrame fusionado (VCF + gnomAD).
    
    Returns:
        None: Muestra el gráfico de dispersión.
    """
    print("\n--- Bloque II E): Comparación Gráfica de Frecuencias ---")
    
    # 1. Definir columnas clave (Nombres confirmados que existen en el DF)
    vcf_af_col = ('INFO', 'AF') 
    gnomad_af_col = ('INFO', 'gnomAD_genome_ALL') 

    # 2. Validación y Selección de Datos
    
    if vcf_af_col not in df.columns or gnomad_af_col not in df.columns:
        print(f"Error: Faltan las columnas necesarias para la comparación: {vcf_af_col} o {gnomad_af_col}.")
        return

    # Creamos el DataFrame de comparación seleccionando SÓLO las dos columnas AF
    comparison_df = df[[vcf_af_col, gnomad_af_col]].copy()
    
    # 3. Conversión de Tipos y Limpieza 
    
    # Aplicamos pd.to_numeric directamente a las columnas de la copia (Solución al ValueError)
    comparison_df[vcf_af_col] = pd.to_numeric(comparison_df[vcf_af_col], errors='coerce')
    comparison_df[gnomad_af_col] = pd.to_numeric(comparison_df[gnomad_af_col], errors='coerce')
    
    # Eliminar cualquier fila donde alguna de las dos AF sea NaN
    comparison_df.dropna(inplace=True) 

    if comparison_df.empty:
        print("No hay variantes con frecuencias alélicas numéricas válidas en ambos datasets para comparar.")
        return

    # 4. Graficar: Scatter Plot
    print(f"Graficando {len(comparison_df)} puntos de datos...")
    
    plt.figure(figsize=(8, 6))
    
    # Línea de identidad (Y=X)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Línea de Identidad (AF_VCF = AF_gnomAD)')
    
    # Graficar los puntos de dispersión 
    plt.scatter(comparison_df[vcf_af_col], comparison_df[gnomad_af_col], 
                alpha=0.6, s=20, label='Variantes coincidentes')
    
    # 5. Configuración del gráfico
    plt.title('Comparación de Frecuencia Alélica (VCF vs. gnomAD)')
    plt.xlabel(f'AF del VCF/Variant Caller (INFO, AF)')
    plt.ylabel(f'AF de gnomAD ({gnomad_af_col})')
    plt.xlim(0, 1) 
    plt.ylim(0, 1)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.show()
    
    print(" Gráfico de comparación de frecuencias generado con éxito.")


# ---- Bloque III: Programa principal ----

def run():
    # --- 1. CONFIGURACIÓN INICIAL ---
    
    # Ruta al VCF Anotado
    vcf_file = "/mnt/c/Backup_compu/INIDEP/CURSOS/Bioinformatica_avanzada/TP7/TP7_anotated.vcf"
    
    # Columna de Anotación (la que se encontró en tu VCF)
    ANNOTATION_COLUMN = ('INFO', 'ANN') 
    
    # Configurar Warnings (para silenciar PerformanceWarning de Pandas)
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    print("\n--- INICIANDO PIPELINE DE ANÁLISIS ---")

    # --- 2. CARGA DEL DATAFRAME (Módulo III) ---
    mi_vcf_df = limpiar_y_cargar_vcf(vcf_file)

    # 3. EJECUCIÓN CONDICIONAL DE MÓDULOS
    if mi_vcf_df is not None:
        
        print("\n--- CARGA EXITOSA ---")
        print(f"Número inicial de variantes: {len(mi_vcf_df)}")

        # MÓDULO A: Análisis de SNVs y Ts/Tv
        ratio_obtenido = analizar_distribucion_snv(mi_vcf_df)
        if ratio_obtenido and ratio_obtenido < 1.5:
            print("\n[ALERTA DE QC] El ratio Ts/Tv está bajo. Considera revisar la calidad de las llamadas.")

        # MÓDULO B: Análisis de Consecuencias Biológicas
        # Asignamos el resultado a mi_vcf_df para añadir las nuevas columnas de categoría
        mi_vcf_df = analizar_consecuencias_biologicas(mi_vcf_df, annotation_column=ANNOTATION_COLUMN)
        
        # MÓDULO C: Análisis de Sustituciones Missense
        # Utiliza las columnas creadas en el Módulo B
        analizar_sustituciones_missense(mi_vcf_df, annotation_column=ANNOTATION_COLUMN)

        # MÓDULO D: Merge con GnomAD (Anotación Externa)
        # Asignamos el resultado a mi_vcf_df para añadir las columnas de gnomAD
        mi_vcf_df = agregar_anotaciones_gnomad(mi_vcf_df)
       
        print("\n--- DIAGNÓSTICO: COLUMNAS MERGEADAS ---")
        print(mi_vcf_df.columns.tolist())

        print("\n--- LISTA COMPLETA DE COLUMNAS DEL DATAFRAME FINAL ---")
        print(mi_vcf_df.columns.tolist())

        print("\n--- DIAGNÓSTICO: COLUMNAS AF ---")

        # Ahora accedemos a la columna real para gnomAD:
        print("Valores de gnomAD AF ('INFO', 'gnomAD_genome_ALL'):")
        print(mi_vcf_df[('INFO', 'gnomAD_genome_ALL')].head(10)) 
        print("Tipo de dato de gnomAD AF:", mi_vcf_df[('INFO', 'gnomAD_genome_ALL')].dtype)

        # La columna VCF AF original sigue siendo:
        print("\nValores de VCF AF ('INFO', 'AF'):")
        print(mi_vcf_df[('INFO', 'AF')].head(10)) 
        print("Tipo de dato de VCF AF:", mi_vcf_df[('INFO', 'AF')].dtype)

        # MÓDULO E: Comparación Gráfica (Visualización)
        # Utiliza el DataFrame fusionado para el gráfico
        comparar_frecuencias_gnomad(mi_vcf_df)

        # --- 4. OUTPUT FINAL ---
        print("\n--- PIPELINE COMPLETADO ---")
        print("Visualiza el gráfico y revisa las columnas añadidas en el DataFrame.")
        print(f"Columnas finales (ejemplo de gnomAD): {list(mi_vcf_df.columns[-3:])}")
    
    else:
        print("\n--- FALLO EN LA EJECUCIÓN ---")
        print("No se pudo iniciar el análisis debido a un error de carga del VCF.")


if __name__ == '__main__':
    run()
