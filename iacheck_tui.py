"""
Interfaz de terminal (TUI) de iacheck, construida con Textual.

    iacheck            # abre la interfaz
    python iacheck_tui.py

Atajos: 1/2/3 pestañas · / buscar · c contexto · o abrir en HF · y copiar comando · r recargar · q salir
"""
from __future__ import annotations

import webbrowser
from typing import Callable, Dict, List, Optional, Tuple

from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (DataTable, Footer, Header, Input, OptionList, Select, Static,
                             TabbedContent, TabPane)
from textual.widgets.option_list import Option

import iacheck as core
from iacheck import fmt_gb, fmt_num, fmt_params

LEVEL_STYLE = {core.LEVEL_GPU: "bold green", core.LEVEL_HYBRID: "bold yellow",
               core.LEVEL_CPU: "bold dark_orange", core.LEVEL_NO: "bold red"}
CONTEXTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
TASKS = {"text-generation": "Texto", "image-text-to-text": "Visión", "any-to-any": "Multimodal",
         "text2text-generation": "Texto", "feature-extraction": "Embeddings",
         "automatic-speech-recognition": "Voz", "text-to-image": "Imagen"}
FILTERS = [("Aptos para mi equipo", "aptos"), ("Solo GPU completa", "3"),
           ("GPU + CPU (offload)", "2"), ("Solo CPU", "1"), ("No aptos", "0"), ("Todos", "todos")]
MODEL_COLUMNS = [("Modelo", "id"), ("Params", "params"), ("Tipo", "task"), ("Descargas", "downloads"),
                 ("Quant", "quant"), ("Memoria", "mem"), ("Velocidad", "speed"), ("Veredicto", "level")]
SORTS: Dict[str, Callable[[core.ModelEval], object]] = {
    "id": lambda e: e.id.lower(),
    "params": lambda e: e.params or 0,
    "task": lambda e: e.task,
    "downloads": lambda e: e.downloads,
    "quant": lambda e: e.quant,
    "mem": lambda e: e.assessment.required_gb if e.assessment else 0,
    "level": lambda e: (e.level, e.params or 0),
    "speed": lambda e: e.assessment.tok_s if e.assessment else 0,
}


SHORT_LABEL = {core.LEVEL_GPU: "● GPU", core.LEVEL_HYBRID: "● GPU+CPU", core.LEVEL_CPU: "● CPU",
               core.LEVEL_NO: "✗ No apto"}


def level_cell(level: int, short: bool = False) -> Text:
    return Text(SHORT_LABEL[level] if short else core.LEVEL_LABEL[level][0], style=LEVEL_STYLE[level])


def bar(frac: float, width: int = 14) -> Text:
    frac = max(0.0, min(1.0, frac))
    n = round(frac * width)
    style = "green" if frac < 0.6 else "yellow" if frac < 0.85 else "red"
    return Text.assemble(("█" * n, style), ("░" * (width - n), "grey35"))


def model_cells(e: core.ModelEval) -> List[Text]:
    task = TASKS.get(e.task, e.task or "—")
    if not e.params or not e.assessment:
        return [Text(core.trunc(e.id, 36)), Text("?", justify="right"), Text(task),
                Text(fmt_num(e.downloads), justify="right"), Text("—"), Text("—"), Text(""),
                Text("? sin datos", style="dim")]
    a = e.assessment
    params = Text(fmt_params(e.params), justify="right")
    if e.active and e.active < e.params * 0.8:
        params.append(f" A{fmt_params(e.active)}", style="dim")
    return [Text(core.trunc(e.id, 36)), params, Text(task), Text(fmt_num(e.downloads), justify="right"),
            Text(e.quant if a.level else "—", style="bold" if a.level else "dim"),
            Text(fmt_gb(a.required_gb), justify="right"),
            Text(f"{a.tok_s:.0f} t/s" if a.level else "", justify="right"), level_cell(a.level, short=True)]


class IACheckApp(App):
    TITLE = "iacheck"
    SUB_TITLE = "¿Qué IA local puede correr mi equipo?"

    CSS = """
    #body { height: 1fr; }
    #sidebar { width: 42; min-width: 36; padding: 0 1; scrollbar-size-vertical: 1; }
    .card { border: round $primary; padding: 0 1; height: auto; margin-bottom: 1; }
    .card:focus-within { border: round $accent; }
    #specs { border: round $accent; }
    #ctx { width: 100%; }
    #tabs { width: 1fr; }
    .toolbar { height: auto; }
    #filter { width: 30; }
    .summary { padding: 1 2; color: $text-muted; width: 1fr; height: auto; }
    DataTable { height: 1fr; }
    #detail-head { height: auto; padding: 0 1; margin-top: 1; }
    #detail-table { height: 1fr; min-height: 6; margin-top: 1; }
    #detail-conclusion { height: auto; padding: 0 1; border: round $success; }
    #gguf-alts { height: auto; max-height: 7; border: round $primary; }
    #gguf-alts.hidden { display: none; }
    """

    BINDINGS = [
        Binding("q", "quit", "Salir"),
        Binding("1", "show_tab('recomendados')", "Recomendados"),
        Binding("2", "show_tab('buscar')", "Buscar"),
        Binding("3", "show_tab('detalle')", "Detalle"),
        Binding("slash", "focus_search", "Buscar", show=False),
        Binding("c", "cycle_ctx", "Contexto"),
        Binding("o", "open_hf", "Abrir en HF"),
        Binding("y", "copy_cmd", "Copiar comando"),
        Binding("r", "reload", "Recargar"),
        Binding("s", "toggle_sidebar", "Panel"),
    ]

    def __init__(self, ctx: int = 4096) -> None:
        super().__init__()
        self.ctx = ctx if ctx in CONTEXTS else 4096
        self.hw: Optional[core.Hardware] = None
        self.ram_avail: Optional[float] = None
        self.vram_used: List[float] = []
        self.candidates: List[dict] = []
        self.online = True
        self.search_query = ""
        self.search_infos: List[dict] = []
        self.detail: Optional[Tuple[dict, Dict[str, float]]] = None
        self.detail_id: Optional[str] = None
        self.detail_cmd: Optional[str] = None
        self.gguf_alts: List[str] = []
        self.filter = "aptos"
        self.sorts: Dict[str, Optional[Tuple[str, bool]]] = {"rec-table": None, "search-table": None}

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with VerticalScroll(id="sidebar"):
                yield Static("Detectando hardware…", id="specs", classes="card")
                yield Static("", id="capacity", classes="card")
                with Vertical(id="ctxbox", classes="card"):
                    yield Select([(f"{n:,} tokens".replace(",", "."), n) for n in CONTEXTS],
                                 value=self.ctx, allow_blank=False, id="ctx")
                yield Static(self.legend(), id="legend", classes="card")
            with TabbedContent(id="tabs", initial="recomendados"):
                with TabPane("Recomendados", id="recomendados"):
                    with Horizontal(classes="toolbar"):
                        yield Select(FILTERS, value="aptos", allow_blank=False, id="filter")
                        yield Static("Consultando Hugging Face…", id="rec-summary", classes="summary")
                    yield DataTable(id="rec-table", cursor_type="row", zebra_stripes=True)
                with TabPane("Buscar", id="buscar"):
                    yield Input(placeholder="Busca un modelo en Hugging Face (llama 3, qwen3 coder, gemma…) y pulsa Enter",
                                id="search-input")
                    yield Static("Escribe un nombre y pulsa Enter. Enter sobre un resultado abre su análisis.",
                                 id="search-summary", classes="summary")
                    yield DataTable(id="search-table", cursor_type="row", zebra_stripes=True)
                with TabPane("Detalle", id="detalle"):
                    yield Input(placeholder="usuario/modelo o URL de Hugging Face y pulsa Enter", id="repo-input")
                    yield Static("Pulsa Enter sobre cualquier modelo de las listas para ver aquí su análisis "
                                 "por cuantización.", id="detail-head")
                    yield DataTable(id="detail-table", cursor_type="row", zebra_stripes=True)
                    yield Static("", id="detail-conclusion")
                    yield OptionList(id="gguf-alts", classes="hidden")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#specs").border_title = "Equipo"
        self.query_one("#capacity").border_title = "Capacidad estimada"
        self.query_one("#ctxbox").border_title = "Contexto (c)"
        self.query_one("#legend").border_title = "Leyenda"
        self.query_one("#detail-conclusion").border_title = "Conclusión"
        self.query_one("#gguf-alts").border_title = "Versiones GGUF (Enter para analizar)"
        for tid in ("rec-table", "search-table"):
            t = self.query_one(f"#{tid}", DataTable)
            for label, key in MODEL_COLUMNS:
                t.add_column(label, key=key)
        dt = self.query_one("#detail-table", DataTable)
        for label in ("", "Cuantización", "Detalle", "Tamaño", "Memoria", "Veredicto", "Velocidad", "Disco"):
            dt.add_column(label)
        self.query_one("#detail-conclusion").display = False
        self.query_one("#rec-table").loading = True
        self.load_all()
        self.set_interval(3, self.live_tick)

    @staticmethod
    def legend() -> Text:
        t = Text()
        for lvl, desc in ((core.LEVEL_GPU, "cabe entero en la GPU"),
                          (core.LEVEL_HYBRID, "reparte entre GPU y RAM"),
                          (core.LEVEL_CPU, "solo cabe en la RAM"),
                          (core.LEVEL_NO, "no cabe")):
            t.append_text(level_cell(lvl))
            t.append(f"\n   {desc}\n", style="dim")
        t.append("\nEnter", style="bold")
        t.append(" ver detalle · ", style="dim")
        t.append("clic en cabecera", style="bold")
        t.append(" ordenar", style="dim")
        return t

    # ---------------------------------------------------------------- workers
    @work(thread=True, exclusive=True, group="load")
    def load_all(self, detect: bool = True) -> None:
        if detect or self.hw is None:
            hw = core.detect_hardware()
            self.call_from_thread(self.set_hardware, hw)
        candidates, online = core.fetch_candidates()
        self.call_from_thread(self.set_candidates, candidates, online)

    @work(thread=True, exclusive=True, group="live")
    def live_tick(self) -> None:
        if self.hw is None:
            return
        _, avail = core._ram()
        used: List[float] = []
        if any(g.vendor == "nvidia" for g in self.hw.gpus):
            out = core._run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], 4)
            used = [float(x) / 1024 for x in out.split() if x.replace(".", "").isdigit()]
        self.call_from_thread(self.set_live, avail, used)

    @work(thread=True, exclusive=True, group="search")
    def search_worker(self, query: str) -> None:
        try:
            results, err = core.search_models(query, 40), None
        except core.HFError as e:
            results, err = [], str(e)
        self.call_from_thread(self.set_search, query, results, err)

    @work(thread=True, exclusive=True, group="detail")
    def detail_worker(self, repo: str) -> None:
        try:
            info, gguf = core.fetch_repo(repo)
            alts = [] if gguf else core.find_gguf_versions(info["id"], 5)
            self.call_from_thread(self.set_detail, info, gguf, alts)
        except core.HFError as e:
            self.call_from_thread(self.detail_error, str(e))

    # -------------------------------------------------------- state -> screen
    def set_hardware(self, hw: core.Hardware) -> None:
        self.hw, self.ram_avail = hw, hw.ram_available_gb
        self.render_specs()
        self.render_capacity()

    def set_live(self, avail: Optional[float], used: List[float]) -> None:
        self.ram_avail, self.vram_used = avail, used
        self.render_specs()

    def set_candidates(self, candidates: List[dict], online: bool) -> None:
        self.candidates, self.online = candidates, online
        self.query_one("#rec-table").loading = False
        if not online:
            self.notify("Sin conexión con Hugging Face: se muestra una lista local de modelos conocidos.",
                        severity="warning", timeout=8)
        self.render_recommended()

    def set_search(self, query: str, results: List[dict], err: Optional[str]) -> None:
        self.query_one("#search-table").loading = False
        self.search_query, self.search_infos = query, results
        if err:
            self.query_one("#search-summary", Static).update(Text(err, style="red"))
        self.render_search()

    def set_detail(self, info: dict, gguf: Dict[str, float], alts: List[str]) -> None:
        self.detail, self.gguf_alts = (info, gguf), alts
        self.query_one("#detail-table").loading = False
        self.render_detail()
        self.query_one("#detail-table").focus()

    def detail_error(self, msg: str) -> None:
        self.query_one("#detail-table").loading = False
        self.query_one("#detail-head", Static).update(Text(msg, style="bold red"))

    def render_specs(self) -> None:
        hw = self.hw
        if hw is None:
            return
        g = Table.grid(padding=(0, 1))
        g.add_column(style="bold cyan", no_wrap=True)
        g.add_column()
        g.add_row("Sistema", f"{hw.os_name} [{hw.arch}]")
        g.add_row("CPU", hw.cpu_name)
        cores = f"{hw.cores_physical} núcleos · {hw.cores_logical} hilos" if hw.cores_physical \
            else f"{hw.cores_logical} hilos"
        g.add_row("", Text(cores, style="dim"))
        g.add_row("RAM", f"{fmt_gb(hw.ram_gb)}")
        if self.ram_avail is not None:
            used = hw.ram_gb - self.ram_avail
            g.add_row("", Text.assemble(bar(used / hw.ram_gb), (f" {used:.0f} GB usados", "dim")))
        if not hw.gpus:
            g.add_row("GPU", Text("Sin GPU compatible (se usará la CPU)", style="yellow"))
        for i, gpu in enumerate(hw.gpus):
            g.add_row("GPU" if len(hw.gpus) == 1 else f"GPU {i + 1}", gpu.name)
            if gpu.unified:
                g.add_row("", Text(f"Memoria unificada: hasta {fmt_gb(gpu.vram_gb)} para la GPU", style="dim"))
            else:
                g.add_row("VRAM", f"{fmt_gb(gpu.vram_gb)} · ~{gpu.bandwidth_gbs:.0f} GB/s")
                nv = [x for x in hw.gpus if x.vendor == "nvidia"]
                if gpu in nv and nv.index(gpu) < len(self.vram_used):
                    used = self.vram_used[nv.index(gpu)]
                    g.add_row("", Text.assemble(bar(used / gpu.vram_gb), (f" {used:.1f} GB usados", "dim")))
        g.add_row("Disco", f"{fmt_gb(hw.disk_free_gb)} libres")
        self.query_one("#specs", Static).update(g)

    def render_capacity(self) -> None:
        if self.hw is None:
            return
        t = Text()
        for level, label, limit, maxes in core.capacity(self.hw, self.ctx):
            t.append("● ", style=LEVEL_STYLE[level])
            t.append(label, style="bold")
            t.append(f"  ({fmt_gb(limit)})\n", style="dim")
            t.append("   " + "  ".join(f"{q.split('_')[0]} ~{fmt_params(p)}" if p >= 3e8 else f"{q.split('_')[0]} —"
                                       for q, p in maxes.items()) + "\n")
        t.append("Tamaño máximo de modelo por cuantización.", style="dim")
        self.query_one("#capacity", Static).update(t)

    def evaluate_all(self, infos: List[dict]) -> List[core.ModelEval]:
        return [core.evaluate(i, self.hw, self.ctx) for i in infos]

    def fill_table(self, table_id: str, evals: List[core.ModelEval]) -> None:
        table = self.query_one(f"#{table_id}", DataTable)
        sort = self.sorts[table_id]
        if sort:
            evals = sorted(evals, key=SORTS[sort[0]], reverse=sort[1])
        table.clear()
        for e in evals:
            table.add_row(*model_cells(e), key=e.id)

    def render_recommended(self) -> None:
        if self.hw is None or not self.candidates:
            return
        evals = [e for e in self.evaluate_all(self.candidates) if e.params and e.params >= 1e8]
        fits = sum(1 for e in evals if e.level > 0)
        if self.filter == "aptos":
            shown = [e for e in evals if e.level > 0]
        elif self.filter == "todos":
            shown = evals
        else:
            shown = [e for e in evals if e.level == int(self.filter)]
        # Por defecto: mejor veredicto primero; en GPU los más grandes, fuera de ella los más rápidos
        shown.sort(key=lambda e: (e.level, (e.params or 0) if e.level == core.LEVEL_GPU
                                  else (e.assessment.tok_s if e.assessment else 0)), reverse=True)
        self.fill_table("rec-table", shown)
        src = "populares y en tendencia en Hugging Face" if self.online else "de la lista local (sin conexión)"
        self.query_one("#rec-summary", Static).update(
            Text.assemble((f"{fits}", "bold green"), f" de {len(evals)} modelos {src} caben en tu equipo"))

    def render_search(self) -> None:
        if self.hw is None or not self.search_query:
            return
        evals = self.evaluate_all(self.search_infos)
        self.fill_table("search-table", evals)
        if self.search_infos:
            ok = sum(1 for e in evals if e.level >= core.LEVEL_HYBRID and e.assessment.tok_s >= 5)
            self.query_one("#search-summary", Static).update(Text.assemble(
                f'Resultados para "{self.search_query}": ', (f"{ok}", "bold green"),
                f" de {len(evals)} corren con fluidez en tu equipo. Enter para ver el detalle."))
        else:
            self.query_one("#search-summary", Static).update(
                Text(f'Sin resultados para "{self.search_query}".', style="yellow"))

    def render_detail(self) -> None:
        if self.hw is None or self.detail is None:
            return
        info, gguf = self.detail
        ra = core.analyze_repo(self.hw, self.ctx, info, gguf)
        ev = ra.ev
        self.detail_id = ev.id

        head = Table.grid(padding=(0, 2))
        head.add_column(style="bold cyan", no_wrap=True)
        head.add_column()
        head.add_row("Modelo", Text(ev.id, style="bold"))
        head.add_row("Enlace", f"{core.HF_WEB}/{ev.id}")
        head.add_row("Tarea", f"{TASKS.get(ev.task, ev.task or 'desconocida')}   ·   "
                              f"{fmt_num(ev.downloads)} descargas   ·   {fmt_num(ev.likes)} likes")
        if ev.params:
            moe = f"   (MoE: ~{fmt_params(ev.active)} activos por token)" \
                if ev.active and ev.active < ev.params * 0.8 else ""
            head.add_row("Parámetros", fmt_params(ev.params) + moe)
        head.add_row("Fuente", "Archivos GGUF del repositorio (tamaño real)" if ra.is_gguf
                     else "Estimación por cuantización")
        if ra.gated:
            head.add_row("Aviso", Text("Acceso restringido: acepta la licencia en Hugging Face antes de descargar.",
                                       style="yellow"))
        self.query_one("#detail-head", Static).update(head)

        table = self.query_one("#detail-table", DataTable)
        table.clear()
        for r in ra.rows:
            a = r.assessment
            star = Text("★", style="bold yellow") if r is ra.chosen and a.level else Text("")
            table.add_row(star, Text(r.name, style="bold" if r is ra.chosen else ""), Text(r.desc, style="dim"),
                          Text(fmt_gb(r.size_gb), justify="right"), Text(fmt_gb(a.required_gb), justify="right"),
                          level_cell(a.level), Text(f"~{a.tok_s:.0f} tok/s" if a.level else "", justify="right"),
                          Text("OK", style="green") if r.disk_ok else Text("sin espacio", style="red"))

        if ra.chosen in ra.rows:
            table.move_cursor(row=ra.rows.index(ra.chosen))

        concl = Text()
        self.detail_cmd = None
        if not ev.params:
            concl.append("No se pudo determinar el tamaño del modelo.", style="yellow")
        elif ra.chosen and ra.chosen.assessment.level > 0:
            a = ra.chosen.assessment
            concl.append_text(level_cell(a.level))
            concl.append("  Recomendado: ")
            concl.append(ra.chosen.name, style="bold")
            concl.append(f"  ·  ~{fmt_gb(a.required_gb)} de memoria  ·  ~{a.tok_s:.0f} tokens/s estimados\n")
            if ra.is_gguf:
                self.detail_cmd = f"ollama run hf.co/{ev.id}:{ra.chosen.name}"
                concl.append("Ollama:    ", style="bold")
                concl.append(self.detail_cmd, style="cyan")
                concl.append("   (y para copiar)\n", style="dim")
                concl.append("LM Studio: ", style="bold")
                concl.append(f"busca {ev.id} y descarga el archivo {ra.chosen.name}")
            else:
                concl.append("Este repositorio no trae GGUF; para Ollama / LM Studio usa una de las versiones "
                             "GGUF de abajo." if self.gguf_alts else
                             "Busca una versión GGUF o MLX de este modelo para usarlo en Ollama o LM Studio.",
                             style="dim")
        else:
            need = min((r.assessment.required_gb for r in ra.rows), default=0)
            concl.append_text(level_cell(core.LEVEL_NO))
            concl.append(f"  Necesita al menos {fmt_gb(need)} incluso cuantizado; "
                         f"tu equipo dispone de ~{fmt_gb(ra.available_gb)}.\n")
            concl.append("Prueba una versión más pequeña de la misma familia.", style="dim")
        box = self.query_one("#detail-conclusion", Static)
        box.update(concl)
        box.display = True

        alts = self.query_one("#gguf-alts", OptionList)
        alts.clear_options()
        alts.add_options([Option(f"{a}") for a in self.gguf_alts])
        alts.set_class(not self.gguf_alts, "hidden")

    def render_all(self) -> None:
        self.render_capacity()
        self.render_recommended()
        self.render_search()
        self.render_detail()

    # ------------------------------------------------------------------ events
    @on(Select.Changed, "#ctx")
    def ctx_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, int) and event.value != self.ctx:
            self.ctx = event.value
            self.render_all()
            self.notify(f"Estimaciones recalculadas para {self.ctx:,} tokens de contexto".replace(",", "."))

    @on(Select.Changed, "#filter")
    def filter_changed(self, event: Select.Changed) -> None:
        self.filter = str(event.value)
        self.render_recommended()

    @on(DataTable.HeaderSelected)
    def sort_by(self, event: DataTable.HeaderSelected) -> None:
        tid = event.data_table.id
        if tid not in self.sorts:
            return
        key = str(event.column_key.value)
        prev = self.sorts[tid]
        reverse = not prev[1] if prev and prev[0] == key else key not in ("id", "task", "quant", "mem")
        self.sorts[tid] = (key, reverse)
        self.render_recommended() if tid == "rec-table" else self.render_search()

    @on(DataTable.RowSelected, "#rec-table, #search-table")
    def row_selected(self, event: DataTable.RowSelected) -> None:
        self.open_detail(str(event.row_key.value))

    @on(Input.Submitted, "#search-input")
    def search_submitted(self, event: Input.Submitted) -> None:
        q = event.value.strip()
        if q:
            self.query_one("#search-table").loading = True
            self.query_one("#search-summary", Static).update(Text(f'Buscando "{q}"…', style="dim"))
            self.search_worker(q)

    @on(Input.Submitted, "#repo-input")
    def repo_submitted(self, event: Input.Submitted) -> None:
        if event.value.strip():
            self.open_detail(event.value.strip())

    @on(OptionList.OptionSelected, "#gguf-alts")
    def alt_selected(self, event: OptionList.OptionSelected) -> None:
        self.open_detail(self.gguf_alts[event.option_index])

    @on(TabbedContent.TabActivated)
    def tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "buscar":
            self.query_one("#search-input").focus()
        elif event.pane.id == "recomendados":
            self.query_one("#rec-table").focus()

    def open_detail(self, repo: str) -> None:
        self.query_one("#tabs", TabbedContent).active = "detalle"
        self.query_one("#repo-input", Input).value = core.normalize_repo(repo)
        self.query_one("#detail-head", Static).update(Text(f"Analizando {repo}…", style="dim"))
        self.query_one("#detail-table").loading = True
        self.query_one("#detail-table").focus()
        self.detail_worker(repo)

    # ----------------------------------------------------------------- actions
    def action_toggle_sidebar(self) -> None:
        sb = self.query_one("#sidebar")
        sb.display = not sb.display

    def on_resize(self, event) -> None:
        # En terminales estrechas se oculta el panel lateral (se muestra con la tecla s)
        self.query_one("#sidebar").display = event.size.width >= 140

    def action_show_tab(self, tab: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab

    def action_focus_search(self) -> None:
        self.action_show_tab("buscar")
        self.query_one("#search-input").focus()

    def action_cycle_ctx(self) -> None:
        nxt = CONTEXTS[(CONTEXTS.index(self.ctx) + 1) % len(CONTEXTS)]
        self.ctx = nxt
        self.query_one("#ctx", Select).value = nxt
        self.render_all()
        self.notify(f"Estimaciones recalculadas para {nxt:,} tokens de contexto".replace(",", "."))

    def action_reload(self) -> None:
        self.query_one("#rec-table").loading = True
        self.query_one("#rec-summary", Static).update("Consultando Hugging Face…")
        self.load_all()

    def current_model(self) -> Optional[str]:
        active = self.query_one("#tabs", TabbedContent).active
        if active == "detalle":
            return self.detail_id
        table = self.query_one("#rec-table" if active == "recomendados" else "#search-table", DataTable)
        if table.row_count == 0:
            return None
        return str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)

    def action_open_hf(self) -> None:
        model = self.current_model()
        if model:
            webbrowser.open(f"{core.HF_WEB}/{model}")
            self.notify(f"Abriendo {model} en el navegador")

    def action_copy_cmd(self) -> None:
        if self.detail_cmd:
            self.copy_to_clipboard(self.detail_cmd)
            self.notify(f"Copiado: {self.detail_cmd}")
        else:
            self.notify("Abre el detalle de un repositorio GGUF para copiar su comando de Ollama.",
                        severity="warning")


def run(ctx: int = 4096) -> None:
    IACheckApp(ctx).run()


if __name__ == "__main__":
    run()
