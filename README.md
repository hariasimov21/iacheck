# iacheck — ¿Qué IA local puede correr mi equipo?

Aplicación de terminal para **Windows, macOS y Linux** que:

1. **Escanea tu equipo**: sistema, CPU, RAM, GPU y VRAM (NVIDIA, AMD, Intel Arc y la memoria unificada de Apple Silicon), además del espacio libre en disco.
2. **Estima la capacidad**: el tamaño máximo de modelo que cabe en GPU, en GPU+RAM y solo en CPU, para cada cuantización (Q4_K_M, Q8_0, FP16…).
3. **Recomienda modelos de Hugging Face**: consulta en vivo los modelos más populares y en tendencia, e indica cuáles puedes correr, con qué cuantización y a qué velocidad aproximada (tokens/s).
4. **Busca cualquier modelo** y te dice si es apto para correr localmente.
5. **Analiza un repositorio concreto**: si tiene archivos GGUF, usa su tamaño real y te da el comando de Ollama listo para copiar.

> **Estado:** probado en Windows 11 con GPU NVIDIA. En macOS y Linux debería funcionar, pero aún no se ha probado en equipos reales: si lo pruebas, abre un *issue* contando cómo te fue.

## Requisitos

- **Python 3.8 o superior** ([python.org](https://www.python.org/downloads/)). En Windows, marca **"Add Python to PATH"** al instalarlo.
- Conexión a internet (para consultar Hugging Face).
- Una terminal moderna: Windows Terminal, Terminal/iTerm2 en macOS o cualquier terminal de Linux.

## Instalación

Directamente desde GitHub, con un solo comando:

```bash
pip install git+https://github.com/hariasimov21/iacheck.git
```

(En macOS usa `pip3`. En Linux, si aparece el error *externally-managed-environment*, usa `pipx install git+https://github.com/hariasimov21/iacheck.git`).

O descargando el código:

```bash
git clone https://github.com/hariasimov21/iacheck.git
cd iacheck
pip install .
```

## Uso

```bash
iacheck
```

Si la terminal dice que no reconoce `iacheck`, la carpeta de scripts de Python no está en el PATH. Puedes abrirlo igualmente con:

```bash
python -m iacheck
```

(en macOS / Linux: `python3 -m iacheck`).

Para actualizar a la última versión:

```bash
pip install --upgrade --force-reinstall git+https://github.com/hariasimov21/iacheck.git
```

Para desinstalar: `pip uninstall iacheck`.

## La interfaz

Escribe `iacheck` y se abre la aplicación a pantalla completa:

- **Panel izquierdo**: especificaciones del equipo con uso de RAM y VRAM en vivo, capacidad estimada por cuantización, selector de contexto y leyenda.
- **Pestaña Recomendados**: los modelos populares y en tendencia de Hugging Face evaluados para tu equipo. Se pueden filtrar por veredicto y ordenar haciendo clic en las cabeceras.
- **Pestaña Buscar**: escribe cualquier nombre y pulsa Enter para ver si cada resultado es apto.
- **Pestaña Detalle**: análisis por cuantización (o por archivo GGUF real) con la opción recomendada ★, el comando de Ollama y versiones GGUF alternativas.

| Tecla        | Acción                                         |
|--------------|------------------------------------------------|
| `1` `2` `3`  | Cambiar de pestaña                             |
| `/`          | Ir a la búsqueda                               |
| `Enter`      | Ver el detalle del modelo seleccionado         |
| `c`          | Cambiar la longitud de contexto                |
| `o`          | Abrir el modelo en Hugging Face                |
| `y`          | Copiar el comando de Ollama                    |
| `s`          | Mostrar/ocultar el panel lateral               |
| `r`          | Recargar la lista de Hugging Face              |
| `q`          | Salir                                          |

También funciona con el ratón. En terminales de menos de 140 columnas el panel lateral se oculta solo (`s` lo muestra).

## Modo comando (para scripts)

```bash
iacheck scan                             # especificaciones + modelos recomendados
iacheck search qwen3                     # buscar modelos y ver si son aptos
iacheck check Qwen/Qwen3-8B              # análisis detallado (acepta también la URL de HF)
iacheck specs                            # solo hardware y capacidad
iacheck menu                             # menú de texto simple, sin interfaz
```

Opciones útiles:

| Opción        | Qué hace                                                        |
|---------------|-----------------------------------------------------------------|
| `--ctx 16384` | Estima con un contexto más largo (el KV-cache ocupa más memoria) |
| `--top 25`    | Muestra más modelos recomendados en `scan`                      |
| `--limit 40`  | Muestra más resultados en `search`                              |
| `--json`      | Salida en JSON (`scan` y `search`)                              |
| `--no-color`  | Sin colores                                                     |

Para modelos privados o con acceso restringido, define la variable `HF_TOKEN` con tu token de Hugging Face.

## Veredictos

| Veredicto              | Significado                                                        |
|------------------------|--------------------------------------------------------------------|
| ● Excelente (GPU)      | El modelo cabe completo en la VRAM / memoria unificada: rápido.    |
| ● Aceptable (GPU+CPU)  | Parte en GPU y parte en RAM (offload). Funciona, pero más lento.   |
| ● Lento (solo CPU)     | Solo cabe en la RAM del sistema. Útil para pruebas.                |
| ✗ No apto              | No cabe ni con la cuantización más agresiva.                       |

## Cómo se estima

- **Memoria necesaria** = pesos (`parámetros × bits por peso / 8`) + ~3 % + KV-cache según el contexto + ~0,5 GB del runtime.
- **VRAM usable**: VRAM total menos ~0,6 GB para el sistema. En Apple Silicon, ~67 % de la RAM (~75 % con más de 36 GB), que es el límite de macOS por defecto.
- **RAM usable**: RAM total menos el 15 % (mínimo 2,5 GB) para el sistema.
- **Velocidad**: la generación está limitada por el ancho de banda de memoria, así que se usa `ancho de banda / tamaño de pesos activos`. Los modelos MoE (p. ej. `30B-A3B`) solo leen los parámetros activos por token, por eso son mucho más rápidos.
- Los números de parámetros se leen de los metadatos de Hugging Face (safetensors / GGUF) o, si faltan, del nombre del modelo.

Son **estimaciones**: la velocidad real depende del runtime (Ollama, LM Studio, llama.cpp, MLX), los drivers y el contexto usado.

## Crear un ejecutable independiente (opcional)

Para distribuirlo sin Python, compílalo en cada sistema operativo con PyInstaller:

```bash
pip install pyinstaller
pyinstaller --onefile --name iacheck --collect-data textual iacheck.py
```

El binario queda en `dist/`.

## Licencia

[MIT](LICENSE) © 2026 hariasimov21
