#!/usr/bin/env python3
"""
IA para consultar MotherDuck en lenguaje natural, todo en un solo archivo.

Modos:
  python motherduck_ia.py                         Chat con modelo local (Ollama), por defecto qwen3.5:4b
  python motherduck_ia.py --modelo gemma4:e4b     Chat con otro modelo local
  python motherduck_ia.py --comparar              Compara modelos con PREGUNTAS_PRUEBA
  python motherduck_ia.py --nativo                Usa la IA integrada de MotherDuck (no necesita Ollama)
  python motherduck_ia.py --gemini                Chat con Gemini (necesita GEMINI_API_KEY)

Opciones:
  --base NOMBRE      base de MotherDuck (por defecto: variable MOTHERDUCK_DB o "my_db")
  --pensar           activa el razonamiento del modelo (más preciso, mucho más lento)
  --modelos A B ...  modelos a comparar con --comparar

En el chat:
  sql <pregunta>     (solo en --nativo) muestra el SQL generado sin ejecutarlo
  salir              termina

Requisitos:
  pip install duckdb ollama google-genai
  export MOTHERDUCK_TOKEN='tu_token'     (idealmente de solo lectura)
  export GEMINI_API_KEY='tu_api_key'     (solo para --gemini)
  ollama pull qwen3.5:4b                 (no hace falta para --nativo)
"""
import argparse
import json
import os
import re
import sys
import time

import duckdb

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
MODELO_POR_DEFECTO = "qwen3.5:4b"
MODELOS_A_COMPARAR = ["qwen3.5:4b", "gemma4:e4b"]
MODELO_GEMINI = "gemini-3.5-flash"   # si falla, prueba "gemini-2.5-flash"
MAX_FILAS = 50        # filas que se le devuelven al modelo o se muestran en pantalla
MAX_PASOS = 8         # iteraciones máximas del agente por pregunta
CONTEXTO = 8192       # num_ctx de Ollama (súbelo si te sobra RAM/VRAM)
MAX_TOKENS = 4096     # tope de tokens por respuesta; corta bucles infinitos

# Reemplaza por preguntas reales sobre tus datos (se usan con --comparar)
PREGUNTAS_PRUEBA = [
    "¿Cuántas filas tiene cada tabla?",
    "¿Qué columnas tiene la tabla con más registros?",
    "Muéstrame 5 filas de ejemplo de la tabla con más registros.",
]

GRIS, NORMAL = "\033[90m", "\033[0m"
con: duckdb.DuckDBPyConnection | None = None


# ---------------------------------------------------------------------------
# Conexión y esquema
# ---------------------------------------------------------------------------
def conectar(base: str) -> duckdb.DuckDBPyConnection:
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        sys.exit("Falta MOTHERDUCK_TOKEN. Ejecuta: export MOTHERDUCK_TOKEN='tu_token'")
    return duckdb.connect(f"md:{base}?motherduck_token={token}")


def obtener_esquema() -> str:
    filas = con.execute("""
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_catalog = current_database()
          AND table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_schema, table_name, ordinal_position
    """).fetchall()

    tablas: dict[str, list[str]] = {}
    for esquema, tabla, columna, tipo in filas:
        tablas.setdefault(f"{esquema}.{tabla}", []).append(f"{columna} {tipo}")
    return "\n".join(f"- {t}({', '.join(cols)})" for t, cols in tablas.items())


def construir_sistema(esquema: str, herramienta: str = "ejecutar_sql") -> str:
    return f"""Eres un analista de datos. Respondes preguntas consultando una base
DuckDB/MotherDuck con la herramienta `{herramienta}`.

Reglas:
- Usa SQL del dialecto DuckDB. Solo consultas de lectura (SELECT / WITH).
- Usa EXACTAMENTE los nombres de tablas y columnas del esquema. No inventes columnas.
- Siempre agrega LIMIT cuando la consulta pueda devolver muchas filas.
- Si una consulta falla, lee el error, corrígela y vuelve a intentar.
- Nunca adivines cifras: consulta la base antes de responder.
- Responde en español, con los números clave y una explicación breve.

Esquema disponible:
{esquema}
"""


# ---------------------------------------------------------------------------
# Herramienta SQL (la que usa el modelo local)
# ---------------------------------------------------------------------------
# Red de seguridad, NO la seguridad real: usa un token de solo lectura.
PERMITIDO = re.compile(r"^\s*(with|select|describe|show|summarize)\b", re.IGNORECASE)


def ejecutar_sql(query: str) -> str:
    sin_punto_final = query.strip().rstrip(";")
    if not PERMITIDO.match(query) or ";" in sin_punto_final:
        return "Error: solo se permite una única consulta de lectura (SELECT/WITH)."
    try:
        cur = con.execute(sin_punto_final)
        columnas = [d[0] for d in cur.description]
        filas = cur.fetchmany(MAX_FILAS + 1)
        return json.dumps(
            {"columnas": columnas, "filas": filas[:MAX_FILAS], "truncado": len(filas) > MAX_FILAS},
            default=str,
            ensure_ascii=False,
        )
    except Exception as e:
        return f"Error de SQL: {e}"


HERRAMIENTAS = [{
    "type": "function",
    "function": {
        "name": "ejecutar_sql",
        "description": (
            "Ejecuta una consulta SQL de solo lectura (dialecto DuckDB) sobre MotherDuck "
            "y devuelve las columnas y filas en JSON."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Una sola sentencia SELECT o WITH en SQL de DuckDB.",
                }
            },
            "required": ["query"],
        },
    },
}]


def atender_llamadas(llamadas, mensajes: list, mostrar: bool) -> tuple[int, int]:
    """Ejecuta las tool calls del modelo y agrega los resultados al historial."""
    consultas = errores = 0
    for llamada in llamadas:
        consultas += 1
        nombre = llamada.function.name
        if nombre == "ejecutar_sql":
            sql = llamada.function.arguments.get("query", "")
            if mostrar:
                print(f"\n  [SQL] {sql}", flush=True)
            salida = ejecutar_sql(sql)
        else:
            salida = f"Error: la herramienta '{nombre}' no existe."
        if salida.startswith("Error"):
            errores += 1
        mensajes.append({"role": "tool", "content": salida, "tool_name": nombre})
    return consultas, errores


# ---------------------------------------------------------------------------
# Modo chat con modelo local (streaming)
# ---------------------------------------------------------------------------
def preguntar_en_vivo(chat, modelo: str, pensar: bool, historial: list) -> None:
    for _ in range(MAX_PASOS):
        stream = chat(
            model=modelo,
            messages=historial,
            tools=HERRAMIENTAS,
            think=pensar,
            stream=True,
            options={"num_ctx": CONTEXTO, "num_predict": MAX_TOKENS},
        )

        pensamiento, contenido, llamadas = "", "", []
        en_pensamiento = False
        for chunk in stream:
            m = chunk.message
            if m.thinking:
                if not en_pensamiento:
                    print(f"{GRIS}[pensando] ", end="", flush=True)
                    en_pensamiento = True
                print(m.thinking, end="", flush=True)
                pensamiento += m.thinking
            if m.content:
                if en_pensamiento:
                    print(f"{NORMAL}\n", flush=True)
                    en_pensamiento = False
                print(m.content, end="", flush=True)
                contenido += m.content
            if m.tool_calls:
                llamadas.extend(m.tool_calls)
            if chunk.done and chunk.done_reason == "length":
                print(f"{NORMAL}\n[aviso] se alcanzó MAX_TOKENS; la respuesta quedó cortada.")
        if en_pensamiento:
            print(NORMAL)

        historial.append({"role": "assistant", "thinking": pensamiento,
                          "content": contenido, "tool_calls": llamadas})
        if not llamadas:
            print()
            return
        atender_llamadas(llamadas, historial, mostrar=True)

    print("\nSe alcanzó el límite de pasos sin una respuesta final.")


def modo_chat(modelo: str, sistema: str, pensar: bool) -> None:
    from ollama import chat

    historial: list = [{"role": "system", "content": sistema}]
    print(f"Modelo local: {modelo}. Escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("Tú: ").strip()
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        if not pregunta:
            continue
        historial.append({"role": "user", "content": pregunta})
        print("\nIA: ", end="", flush=True)
        try:
            preguntar_en_vivo(chat, modelo, pensar, historial)
        except ConnectionError:
            print("\nNo hay conexión con Ollama. Revisa: systemctl status ollama")
        except Exception as e:
            print(f"\nError del modelo: {e}")
        print()


# ---------------------------------------------------------------------------
# Modo comparar modelos
# ---------------------------------------------------------------------------
def correr_sin_stream(chat, modelo: str, pregunta: str, sistema: str, pensar: bool) -> dict:
    mensajes = [{"role": "system", "content": sistema}, {"role": "user", "content": pregunta}]
    consultas = errores = 0
    inicio = time.perf_counter()
    try:
        for _ in range(MAX_PASOS):
            resp = chat(model=modelo, messages=mensajes, tools=HERRAMIENTAS, think=pensar,
                        options={"num_ctx": CONTEXTO, "num_predict": MAX_TOKENS})
            m = resp.message
            mensajes.append(m)
            if not m.tool_calls:
                return {"ok": True, "respuesta": m.content or "", "consultas": consultas,
                        "errores": errores, "segundos": time.perf_counter() - inicio}
            c, e = atender_llamadas(m.tool_calls, mensajes, mostrar=False)
            consultas += c
            errores += e
        respuesta = "(alcanzó el límite de pasos sin responder)"
    except Exception as e:
        respuesta = f"(falló: {e})"
    return {"ok": False, "respuesta": respuesta, "consultas": consultas,
            "errores": errores, "segundos": time.perf_counter() - inicio}


def modo_comparar(modelos: list[str], sistema: str, pensar: bool) -> None:
    from ollama import chat, generate

    resumen = {}
    for modelo in modelos:
        print(f"\n{'=' * 70}\nModelo: {modelo}\n{'=' * 70}")
        try:  # cargar el modelo antes de medir, para no contar el tiempo de carga
            chat(model=modelo, messages=[{"role": "user", "content": "hola"}], think=False,
                 options={"num_ctx": CONTEXTO, "num_predict": 1})
        except Exception as e:
            print(f"No se pudo cargar {modelo}: {e}")
            continue

        resultados = []
        for pregunta in PREGUNTAS_PRUEBA:
            r = correr_sin_stream(chat, modelo, pregunta, sistema, pensar)
            resultados.append(r)
            estado = "OK" if r["ok"] else "FALLÓ"
            alerta = "  <- respondió sin consultar la base" if r["ok"] and r["consultas"] == 0 else ""
            print(f"\nP: {pregunta}")
            print(f"   [{estado}] {r['segundos']:.1f}s | consultas: {r['consultas']} "
                  f"| errores SQL: {r['errores']}{alerta}")
            texto = r["respuesta"].strip().replace("\n", " ")
            print(f"   R: {texto[:300]}{'...' if len(texto) > 300 else ''}")

        resumen[modelo] = resultados
        generate(model=modelo, prompt="", keep_alive=0)  # liberar memoria

    print(f"\n{'=' * 70}\nRESUMEN\n{'=' * 70}")
    print(f"{'Modelo':<16}{'Respondidas':>12}{'Tiempo total':>14}{'Promedio':>10}"
          f"{'Consultas':>11}{'Errores':>9}")
    for modelo, rs in resumen.items():
        total = sum(r["segundos"] for r in rs)
        print(f"{modelo:<16}{sum(r['ok'] for r in rs):>7}/{len(rs):<4}{total:>13.1f}s"
              f"{total / len(rs):>9.1f}s{sum(r['consultas'] for r in rs):>11}"
              f"{sum(r['errores'] for r in rs):>9}")


# ---------------------------------------------------------------------------
# Modo Gemini (API de Google)
# ---------------------------------------------------------------------------
def consultar_sql(query: str) -> str:
    """Ejecuta una consulta SQL de solo lectura (dialecto DuckDB) sobre MotherDuck.

    Args:
        query: Una sola sentencia SELECT o WITH en SQL de DuckDB.

    Returns:
        JSON con las columnas y filas del resultado, o un mensaje de error.
    """
    print(f"\n  [SQL] {query}", flush=True)
    return ejecutar_sql(query)


def modo_gemini(modelo: str, sistema: str) -> None:
    from google import genai
    from google.genai import types

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("Falta GEMINI_API_KEY. Agrégala a tu archivo .env")

    cliente = genai.Client()  # lee GEMINI_API_KEY del entorno
    chat = cliente.chats.create(
        model=modelo,
        config=types.GenerateContentConfig(
            system_instruction=sistema,
            tools=[consultar_sql],  # el SDK ejecuta la función automáticamente
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=max(MAX_PASOS, 1),
            ),
            max_output_tokens=MAX_TOKENS if MAX_TOKENS > 0 else None,
        ),
    )

    print(f"Gemini: {modelo}. Escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("Tú: ").strip()
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        if not pregunta:
            continue
        try:
            resp = chat.send_message(pregunta)
            print(f"\nIA: {resp.text or '(Gemini no devolvió texto; intenta reformular)'}\n")
        except Exception as e:
            print(f"\nError de Gemini: {e}\n")


# ---------------------------------------------------------------------------
# Modo nativo: IA integrada de MotherDuck (prompt_query / prompt_sql)
# ---------------------------------------------------------------------------
def imprimir_tabla(columnas: list[str], filas: list[tuple]) -> None:
    textos = [["" if v is None else str(v) for v in fila] for fila in filas]
    anchos = [max([len(c)] + [len(f[i]) for f in textos]) for i, c in enumerate(columnas)]

    def linea(valores: list[str]) -> str:
        return " | ".join(v.ljust(a) for v, a in zip(valores, anchos))

    print(linea(columnas))
    print("-+-".join("-" * a for a in anchos))
    for fila in textos:
        print(linea(fila))


def modo_nativo() -> None:
    print("IA integrada de MotherDuck. Usa 'sql <pregunta>' para ver solo el SQL. "
          "Escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("Tú: ").strip()
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        if not pregunta:
            continue
        print()
        try:
            if pregunta.lower().startswith("sql "):
                texto = pregunta[4:].strip().replace("'", "''")
                print(con.execute(f"CALL prompt_sql('{texto}')").fetchone()[0])
            else:
                texto = pregunta.replace("'", "''")
                cur = con.execute(f"PRAGMA prompt_query('{texto}')")
                if cur.description is None:
                    print("(la consulta no devolvió resultados)")
                else:
                    filas = cur.fetchmany(MAX_FILAS + 1)
                    imprimir_tabla([d[0] for d in cur.description], filas[:MAX_FILAS])
                    if len(filas) > MAX_FILAS:
                        print(f"... (mostrando solo las primeras {MAX_FILAS} filas)")
        except Exception as e:
            print(f"Error: {e}")
        print()


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def main() -> None:
    global con

    p = argparse.ArgumentParser(description="IA para consultar MotherDuck en lenguaje natural")
    p.add_argument("--modelo", default=None,
                   help=f"modelo a usar (por defecto {MODELO_POR_DEFECTO}, o {MODELO_GEMINI} con --gemini)")
    p.add_argument("--base", default=os.environ.get("MOTHERDUCK_DB", "my_db"),
                   help="base de datos en MotherDuck")
    p.add_argument("--pensar", action="store_true", help="activar razonamiento (más lento)")
    p.add_argument("--modelos", nargs="+", default=MODELOS_A_COMPARAR,
                   help="modelos a comparar con --comparar")
    modo = p.add_mutually_exclusive_group()
    modo.add_argument("--comparar", action="store_true", help="comparar modelos locales")
    modo.add_argument("--nativo", action="store_true", help="usar la IA integrada de MotherDuck")
    modo.add_argument("--gemini", action="store_true", help="usar Gemini (API de Google)")
    args = p.parse_args()

    con = conectar(args.base)
    print(f"Conectado a md:{args.base}")

    if args.nativo:
        modo_nativo()
        return

    esquema = obtener_esquema()
    if args.gemini:
        modo_gemini(args.modelo or MODELO_GEMINI, construir_sistema(esquema, "consultar_sql"))
    elif args.comparar:
        modo_comparar(args.modelos, construir_sistema(esquema), args.pensar)
    else:
        modo_chat(args.modelo or MODELO_POR_DEFECTO, construir_sistema(esquema), args.pensar)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nHasta luego.")
