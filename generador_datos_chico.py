import pandas as pd
import random
import datetime
import json
import os
import argparse
import sys

# --- CONFIGURACIÓN ---
OBSERVACIONES_POSITIVAS = [
    "Excelente participación en clases.", "Demuestra gran interés por la asignatura.",
    "Colabora activamente con sus compañeros.", "Muy responsable y puntual con sus entregas.",
    "Creativo y con gran potencial.", "Ha mejorado notablemente su rendimiento.",
    "Liderazgo positivo dentro del grupo."
]
OBSERVACIONES_NEGATIVAS = [
    "Falta de estudio y preparación para las evaluaciones.", "Dificultad para concentrarse en clases.",
    "Progreso inconsistente, alterna entre notas altas y bajas.",
    "Molesta a sus compañeros e interrumpe la clase.", "Falta de respeto hacia el profesor.",
    "Se distrae con facilidad.", "Tímido, le cuesta participar oralmente.",
    "Registra agresiones físicas a compañeros.", "Copia en la prueba."
]
ENTREVISTAS_EJEMPLOS = [
    "Apoderado: Se observa con los apoderados una mejora en la intención de estudios.",
    "Apoderado: Padres preocupados porque se duerme muy tarde jugando en el computador.",
    "Apoderado: Padres manifiestan que están viviendo situación de separación que ha afectado a sus hijos.",
    "Los apoderados informan que están en un proceso de separación y que eso ha afectado a sus hijos.",
    "Los apoderados agradecen la actividad pastoral de la semana pasada.",
    "El colegio hace saber a los apoderados la preocupación por el mal comportamiento de su hijo.",
    "Los apoderados solicitan al colegio que su hijo sea incorporado a la selección de futbol.",
    "Se conversa con apoderado sobre la importancia de reforzar hábitos de estudio en casa.",
    "Apoderado justifica inasistencias por viaje familiar, presenta justificativo médico.",
    "Se felicita al apoderado por el notable avance del estudiante en conducta y responsabilidad.",
    "Apoderado informa que el estudiante presenta dificultades de concentración en casa y solicita estrategias de apoyo.",
    "Se cita a apoderado por reiteradas faltas de respeto a inspectores. Apoderado se compromete a conversar con su hijo.",
    "Apoderado informa diagnóstico reciente de TDAH del estudiante. Se acuerda derivar a equipo PIE para evaluación de apoyos.",
    "Se destaca al apoderado el excelente desempeño del estudiante en el debate de Historia, mostrando gran liderazgo.",
    "Apoderado consulta sobre el proceso de postulación a becas para el próximo año. Se entrega información y fechas relevantes.",
    "Conversación sobre la integración social del estudiante. Apoderados mencionan que es tímido; se acuerda fomentar participación en clases.",
    "Se informa al apoderado sobre la falta recurrente de entrega de tareas. Apoderado se compromete a revisar la agenda escolar diariamente.",
    "Apoderados expresan preocupación por posible situación de acoso escolar. Convivencia Escolar iniciará protocolo de investigación.",
    "Apoderado solicita reunión con UTP para revisar la cobertura curricular y los métodos de evaluación de la asignatura de Inglés.",
    "Reunión de seguimiento con apoderado. Se revisan los compromisos de la entrevista anterior y se constatan avances positivos en la conducta."
]
PROBABILIDAD_ENTREVISTA = 0.3
PROBABILIDAD_OBSERVACION = 0.4
RANGO_NOTAS = (2.0, 7.0)
RANGO_ASISTENCIA = (0.85, 1.0)
MESES = ["marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

def cargar_roster_desde_csv(csv_path):
    """Carga los estudiantes únicos desde un CSV existente."""
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el archivo semilla '{csv_path}'")
        sys.exit(1)
    
    print(f"Cargando estudiantes base desde '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
    except Exception:
        # Intentar con separador coma si falla punto y coma
        df = pd.read_csv(csv_path, sep=',', encoding='utf-8-sig')

    # Identificar columnas clave
    cols = df.columns
    # Normalizar nombres de columnas para facilitar búsqueda
    norm_cols = {c.lower().strip(): c for c in cols}
    
    col_id = norm_cols.get('id estudiante') or norm_cols.get('id')
    col_nombre = norm_cols.get('nombre')
    col_curso = norm_cols.get('curso')
    col_edad = norm_cols.get('edad')
    col_profesor = norm_cols.get('profesor')
    col_familia = norm_cols.get('familia')
    col_debiles = norm_cols.get('materias_debiles')
    col_asig = norm_cols.get('asignatura')

    if not (col_id and col_nombre):
        print("Error: El CSV semilla debe tener columnas 'ID Estudiante' y 'Nombre'.")
        sys.exit(1)

    # Agrupar por estudiante para obtener su perfil único
    estudiantes = []
    grupos = df.groupby(col_id)
    
    for eid, group in grupos:
        row = group.iloc[0]
        nombre = row[col_nombre]
        curso = row[col_curso] if col_curso else ""
        edad = row[col_edad] if col_edad else 0
        profesor = row[col_profesor] if col_profesor else ""
        familia = row[col_familia] if col_familia else ""
        debiles = row[col_debiles] if col_debiles else ""
        
        # Obtener lista de asignaturas que cursa este estudiante
        asignaturas = group[col_asig].unique().tolist() if col_asig else []
        
        # Calcular un promedio objetivo aproximado basado en sus notas actuales
        # para mantener coherencia en su rendimiento futuro
        promedio_obj = 5.5
        if norm_cols.get('nota'):
             notas = pd.to_numeric(group[norm_cols['nota']], errors='coerce').dropna()
             if not notas.empty:
                 promedio_obj = notas.mean()

        estudiantes.append({
            "id_estudiante": eid,
            "nombre": nombre,
            "curso": curso,
            "edad": edad,
            "asignaturas": asignaturas,
            "materias_debiles": debiles,
            "profesor": profesor,
            "Familia": familia,
            "promedio_objetivo": promedio_obj
        })
    
    print(f"Se identificaron {len(estudiantes)} estudiantes únicos.")
    return estudiantes

def cargar_cierre_ano_anterior(csv_prev_path, estudiantes):
    """
    Lee el archivo de Diciembre del año anterior y ajusta el 'promedio_objetivo'
    de cada estudiante según cómo terminó el año.
    """
    if not os.path.exists(csv_prev_path):
        print(f"Advertencia: Archivo de cierre anterior '{csv_prev_path}' no existe. Se usará base inicial.")
        return estudiantes
    
    print(f"Cargando historial de cierre desde '{csv_prev_path}' para ajustar Year 2...")
    try:
        df = pd.read_csv(csv_prev_path, sep=';', encoding='utf-8-sig')
    except:
        df = pd.read_csv(csv_prev_path, sep=',', encoding='utf-8-sig')
        
    # Normalizar columnas
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Calcular promedio final real por alumno en ese archivo
    # Asumimos columna 'id estudiante' y 'nota'
    if 'id estudiante' not in df.columns or 'nota' not in df.columns:
        return estudiantes
        
    df['nota'] = pd.to_numeric(df['nota'], errors='coerce')
    promedios_cierre = df.groupby('id estudiante')['nota'].mean().to_dict()
    
    impactados = 0
    for est in estudiantes:
        eid = est['id_estudiante']
        if eid in promedios_cierre:
            cierre = promedios_cierre[eid]
            
            # LÓGICA DE HERENCIA:
            # 1. Si cerró rojo (<4.0): Parte el nuevo año con desventaja severa (-0.5 a su base)
            #    Simula que "no aprendió la base" del año anterior.
            if cierre < 4.0:
                est['promedio_objetivo'] = max(2.0, cierre - 0.3)
                est['etiqueta_simulacion'] = 'riesgo_heredado'
                
            # 2. Si cerró mediocre (4.0 - 5.0): Parte igual o levemente peor
            elif cierre < 5.0:
                est['promedio_objetivo'] = cierre
                
            # 3. Si cerró excelente (>6.0): Mantiene su excelencia
            else:
                est['promedio_objetivo'] = cierre
            
            impactados += 1
            
    print(f"Se ajustaron los perfiles de {impactados} estudiantes basados en su cierre anterior.")
    return estudiantes

def generar_mes(mes_idx, estudiantes, semilla_csv, suffix_ano=""):
    """Genera los datos para un mes específico."""
    nombre_mes = MESES[mes_idx]
    
    # Nombre de archivo puede incluir sufijo si es año 2 (ej. datos_marzo_y2.csv)
    out_csv = f"datos_{nombre_mes}{suffix_ano}.csv"
    print(f"Generando datos para: {nombre_mes.capitalize()} -> {out_csv}")
    
    datos_out = []
    
    # Variabilidad mensual global (ej. en marzo notas más altas, en diciembre cansancio)
    factor_mes = 0.0
    if mes_idx == 0: factor_mes = 0.2 # Marzo empieza bien
    if mes_idx == 9: factor_mes = -0.2 # Diciembre cansados

    for est in estudiantes:
        # Variación de asistencia del mes
        # Base alta (0.85-1.0) pero con fluctuación mensual
        asistencia_mes = round(random.uniform(RANGO_ASISTENCIA[0], RANGO_ASISTENCIA[1]), 2)
        # Algunos alumnos bajan asistencia drasticamente a veces
        if random.random() < 0.05:
            asistencia_mes = round(random.uniform(0.6, 0.8), 2)
            
        # Entrevistas: Evento mensual
        entrevista_mes = "Sin entrevistas registradas."
        if random.random() < PROBABILIDAD_ENTREVISTA:
             entrevista_mes = random.choice(ENTREVISTAS_EJEMPLOS)
        
        promedio_base = est["promedio_objetivo"]
        
        # Iterar asignaturas
        registros_asignaturas = []
        for asig in est["asignaturas"]:
            # Nota varía alrededor de su promedio + factor mes + aleatorio
            variacion = random.uniform(-0.8, 0.8)
            nota = promedio_base + variacion + factor_mes
            
            # Penalizar materias débiles
            if est["materias_debiles"] and isinstance(est["materias_debiles"], str) and est["materias_debiles"] in asig:
                nota -= 0.8
            
            # Ajustar rango
            nota = max(RANGO_NOTAS[0], min(RANGO_NOTAS[1], nota))
            nota = round(nota, 1)
            
            # Observación mensual
            obs = ""
            if random.random() < PROBABILIDAD_OBSERVACION:
                if nota < 4.0:
                    obs = random.choice(OBSERVACIONES_NEGATIVAS)
                elif nota > 6.0:
                    obs = random.choice(OBSERVACIONES_POSITIVAS)
                else:
                    # Mixto
                    obs = random.choice(OBSERVACIONES_POSITIVAS if random.random() > 0.5 else OBSERVACIONES_NEGATIVAS)
            
            registros_asignaturas.append({
                "Asignatura": asig,
                "Nota": nota,
                "Observacion de conducta": obs
            })
            
        # Consistencia: evitar que un alumno de promedio 6.5 tenga puros rojos
        # (Lógica simplificada del script original mantenida)
        notas = [r["Nota"] for r in registros_asignaturas]
        prom_real = sum(notas)/len(notas) if notas else 0
        
        # Guardar registros
        for r in registros_asignaturas:
            datos_out.append({
                "ID Estudiante": est["id_estudiante"],
                "Nombre": est["nombre"],
                "curso": est["curso"],
                "edad": est["edad"],
                "Asignatura": r["Asignatura"],
                "Nota": r["Nota"],
                "Observacion de conducta": r["Observacion de conducta"],
                "materias_debiles": est["materias_debiles"],
                "Asistencia": asistencia_mes,
                "profesor": est["profesor"],
                "Familia": est["Familia"],
                "Entrevistas": entrevista_mes
            })

    # Guardar CSV
    df_out = pd.DataFrame(datos_out)
    # Ordenar columnas
    cols_order = [
        "ID Estudiante", "Nombre", "curso", "edad", "Asignatura", "Nota",
        "Observacion de conducta", "materias_debiles", "Asistencia",
        "profesor", "Familia", "Entrevistas"
    ]
    # Asegurar que existan todas
    for c in cols_order:
        if c not in df_out.columns:
            df_out[c] = ""
            
    df_out = df_out[cols_order]
    df_out.to_csv(out_csv, index=False, sep=';', encoding='utf-8-sig')
    print(f"Archivo {out_csv} generado con {len(df_out)} registros.")

def main():
    parser = argparse.ArgumentParser(description="Generador secuencial de datos mensuales para estudiantes.")
    parser.add_argument("--semilla", type=str, default="datos_completos_100_estudiantes.csv", help="Archivo CSV base con los estudiantes")
    parser.add_argument("--mes", type=str, default=None, help="Nombre del mes a generar (ej: marzo). Si no se especifica, genera todos secuencialmente.")
    parser.add_argument("--prev-year-closure", type=str, default=None, help="Archivo CSV de diciembre del año anterior para heredar rendimiento.")
    parser.add_argument("--suffix", type=str, default="", help="Sufijo para los archivos generados (ej: _y2)")
    
    args = parser.parse_args()

    estudiantes = cargar_roster_desde_csv(args.semilla)
    
    # Si se provee cierre anterior, ajustar promedios objetivos
    if args.prev_year_closure:
        estudiantes = cargar_cierre_ano_anterior(args.prev_year_closure, estudiantes)
    
    if args.mes:
        m = args.mes.lower()
        if m in MESES:
            idx = MESES.index(m)
            generar_mes(idx, estudiantes, args.semilla, suffix_ano=args.suffix)
        else:
            print(f"Mes '{m}' no válido. Use: {', '.join(MESES)}")
    else:
        print("Generando ciclo completo (Marzo a Diciembre)...")
        for i in range(10):
            generar_mes(i, estudiantes, args.semilla, suffix_ano=args.suffix)

if __name__ == "__main__":
    main()
