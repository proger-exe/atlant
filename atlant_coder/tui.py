"""A quiet conversation over the detached Atlant run service."""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import OptionList, RichLog, Static, TextArea

# Shared semantic colors for Textual chrome and Rich transcript/status content.
COLORS = {
    "surface": "#262624",
    "input": "#30302e",
    "text": "#e8e6dc",
    "muted": "#aaa9a2",
    "subtle": "#99988f",
    "accent": "#df9c7e",
    "identity": "#38d5bd",
    "path": "#64b9ee",
    "model": "#c38ddb",
    "cache": "#7fbf68",
    "mode": "#e5b65c",
    "warning": "#e8897d",
}

FINAL_STATUSES = {"completed", "failed", "stopped", "unverified", "interrupted"}
STATUS = {
    "idle": "Ожидание",
    "starting": "Думает",
    "running": "Выполняет",
    "stopping": "Останавливаюсь",
    "waiting_approval": "Ждёт ответа",
    "waiting_input": "Ждёт уточнения",
    "completed": "Готово",
    "failed": "Ошибка",
    "stopped": "Прерван",
    "unverified": "Завершён без проверки",
    "interrupted": "Прерван",
}
STAGES = {
    "planning": "Изучаю проект",
    "coordinator_started": "Проверяю план",
    "contract_approved": "План готов",
    "generating": "Пишу код",
    "replan_requested": "Уточняю план",
    "candidate_plan_ready": "Проверяю план",
    "post_test_repair": "Исправляю результат проверок",
    "project_verified": "Проверки пройдены",
}
USAGE_LABELS = {
    "input_tokens": "Входные токены",
    "output_tokens": "Выходные токены",
    "cached_input_tokens": "Из кэша",
    "plan_corrections": "Коррекции плана",
    "architecture_reworks": "Переработки архитектуры",
    "post_test_repairs": "Исправления после тестов",
    "draft_reviews": "Проверки черновиков",
}
HELP = """/new                 Новая беседа
/mode agent|generate Работа с проектом / генерация
/runs                Прошлые запуски; /attach номер или ID
/stop · /resume       Остановить / продолжить запуск
/retry               Повторить задачу новым запуском
/diff · /checks       Изменения / проверки
/plan · /details      План / причина завершения
/agents · /usage      Агенты / токены и исправления
/allow · /deny        Ответить на запрос разрешения
/quit                Закрыть интерфейс, оставив работу в фоне

Enter отправить · Alt+Enter перенос · Esc / Ctrl+C остановить
Shift+Tab режим · Ctrl+R запуски · Ctrl+Q выход
PgUp/PgDn история · Ctrl+L ввод · Tab запрос разрешения
В запросе: ↑↓ выбрать · Enter подтвердить · Tab вернуться к вводу"""


class Prompt(TextArea):
    BINDINGS = [
        Binding("enter", "submit", show=False, priority=True),
        Binding("alt+enter,shift+enter", "newline", show=False, priority=True),
    ]

    class Submitted(Message):
        pass

    def action_submit(self) -> None:
        self.post_message(self.Submitted())

    def action_newline(self) -> None:
        self.insert("\n")


class AtlantApp(App[None]):
    TITLE = "Atlant"
    ENABLE_COMMAND_PALETTE = False
    CSS = (
        "\n".join(f"$atlant-{role}: {value};" for role, value in COLORS.items())
        + """
    Screen { background: $atlant-surface; color: $atlant-text; }
    #conversation { width: 100%; height: 1fr; padding: 0 1; }
    #brand { height: 2; color: $atlant-muted; padding-top: 1; }
    #chat { height: 1fr; background: transparent; overflow-x: hidden; scrollbar-size: 1 1; }
    #status { height: 1; color: $atlant-muted; }
    #plan { display: none; height: auto; max-height: 7; margin-top: 1; }
    #plan.visible { display: block; }
    #activity { height: 1; margin-top: 1; }
    #input { height: auto; border: solid $atlant-subtle; background: $atlant-input; }
    #input:focus-within { border: solid $atlant-muted; }
    #prompt-mark { width: 2; height: 1; background: $atlant-input; }
    #task { height: 1; max-height: 5; border: none; background: $atlant-input; padding: 0; }
    #task:focus { border: none; }
    #hint { height: 1; color: $atlant-subtle; margin-bottom: 1; }
    #approval { display: none; height: 2; border: none; padding: 0; background: $atlant-surface; }
    #approval.visible { display: block; }
    #approval > .option-list--option-highlighted { background: $atlant-input; color: $atlant-text; text-style: bold; }
    """
    )
    BINDINGS = [
        Binding("ctrl+enter", "send", show=False),
        Binding("ctrl+r", "runs", show=False),
        Binding("escape,ctrl+c,ctrl+s", "stop_run", show=False, priority=True),
        Binding("shift+tab", "cycle_mode", show=False, priority=True),
        Binding("ctrl+q", "quit", show=False, priority=True),
        Binding("pageup", "history_page(-1)", show=False, priority=True),
        Binding("pagedown", "history_page(1)", show=False, priority=True),
        Binding("ctrl+l", "focus_prompt", show=False, priority=True),
        Binding("tab", "focus_approval", show=False, priority=True),
    ]

    def __init__(self, client: Any, workspace: str, run_id: str | None = None) -> None:
        super().__init__()
        self.client, self.workspace, self.run_id = client, workspace, run_id
        self.event_seq = 0
        self._busy = False
        self._starting = False
        self._stop_requested = False
        self._stop_inflight = False
        self._polling = False
        self._poll_timer = None
        self._approval_id: str | None = None
        self._approval_inflight = False
        self._status = "idle"
        self._mode = "agent"
        self._snapshot: dict[str, Any] = {}
        self._runs: list[dict] = []
        self._local_messages: list[str] = []
        self._last_activity = ""
        self._reported_result: tuple | None = None
        self._activity_key = None
        self._activity_since = time.monotonic()
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="atlant-tui"
        )
        # Stop must not queue behind a slow send and a slow poll.
        self._control_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="atlant-stop"
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="conversation"):
            yield Static(
                Text(f"atlant  /  {Path(self.workspace).name or self.workspace}"),
                id="brand",
            )
            yield RichLog(
                id="chat", min_width=1, wrap=True, highlight=False, markup=False
            )
            yield Static("", id="plan")
            yield Static("", id="activity")
            yield OptionList(
                "Разрешить один раз", "Отклонить", id="approval", compact=True
            )
            with Horizontal(id="input"):
                yield Static("›", id="prompt-mark")
                yield Prompt(
                    id="task",
                    highlight_cursor_line=False,
                    placeholder="Что нужно сделать?",
                )
            yield Static("", id="status")
            yield Static(
                "enter отправить · alt+enter новая строка · / команды · ctrl+q выход",
                id="hint",
            )

    def on_mount(self) -> None:
        self._render_snapshot({"status": "idle"})
        self._welcome()
        self.query_one("#task", Prompt).focus()
        self.set_interval(1, self._render_status)
        if self.run_id:
            self.run_worker(self._attach(self.run_id))
        if not self.is_headless:
            self._poll_timer = self.set_interval(0.7, self.poll_run)

    def on_unmount(self) -> None:
        if self._poll_timer:
            self._poll_timer.pause()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._control_executor.shutdown(wait=False, cancel_futures=True)

    def _welcome(self) -> None:
        self._chat(
            "Atlant",
            "Опиши задачу — начнём.\nПрошлые запуски: /runs. Все команды: /help.",
        )

    def _chat(self, author: str, message: str) -> None:
        if message and self.query("#chat"):
            log = self.query_one("#chat", RichLog)
            if author == "Вы":
                text = Text()
                text.append("› ", style="bold " + COLORS["accent"])
                text.append(message, style=COLORS["text"])
                log.write(text)
            else:
                text = Text("• ", style=COLORS["accent"])
                if author != "Atlant":
                    text.append(author + "\n", style="bold")
                text.append(message, style=COLORS["text"])
                log.write(text)
            log.write("")

    async def _guarded(
        self, method: str, *args: Any, control: bool = False
    ) -> Any | None:
        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._control_executor if control else self._executor,
                getattr(self.client, method),
                *args,
            )
        except Exception as exc:
            self._chat("Ошибка", str(exc))
            return None

    def on_prompt_submitted(self) -> None:
        self.action_send()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        event.text_area.styles.height = min(
            5, max(1, len(event.text_area.text.splitlines()))
        )

    def action_send(self) -> None:
        self.run_worker(self._send(), group="send")

    async def _send(self) -> None:
        prompt = self.query_one("#task", Prompt)
        text = prompt.text.strip()
        if text == "/stop":
            prompt.clear()
            self.action_stop_run()
            return
        if not text or self._busy:
            return
        if text == "/retry":
            if self._status not in FINAL_STATUSES:
                self._chat("Atlant", "Повторить можно после завершения запуска.")
                return
            text = self._snapshot.get("task", "").strip()
            if not text:
                self._chat("Atlant", "Текст задачи недоступен. Отправь её ещё раз.")
                return
        self._busy = True
        prompt.clear()
        try:
            if text.startswith("/"):
                await self._command(text)
                return
            self._chat("Вы", text)
            if self.run_id and self._status not in FINAL_STATUSES:
                result = await self._guarded("steer", self.run_id, text)
            else:
                self._starting = True
                self._stop_requested = False
                self._last_activity = ""
                self._render_snapshot({"status": "starting", "mode": self._mode})
                result = await self._guarded("start", self.workspace, text, self._mode)
                self._starting = False
                if result:
                    self.run_id = str(result["id"])
                    self.event_seq = 0
                    self._reported_result = None
                    if self._stop_requested:
                        await self._request_stop()
            if result is None:
                # Keep a failed submission available without replacing text typed while waiting.
                if not prompt.text:
                    prompt.load_text(text)
                return
            self._local_messages.append(text)
            snapshot = await self._guarded("snapshot", self.run_id)
            if snapshot:
                self._render_snapshot(snapshot)
        finally:
            self._starting = False
            self._busy = False

    async def _command(self, text: str) -> None:
        command, _, arg = text.partition(" ")
        arg = arg.strip()
        if command in {"/", "/help"}:
            self._chat("Команды", HELP)
        elif command == "/quit":
            self.exit()
        elif command == "/plan":
            self._chat("План", self._plan_text().plain or "План ещё не составлен.")
        elif command == "/details":
            self._chat(
                "Подробнее",
                self._result_reason() + "\nПроверки: /checks. Изменения: /diff.",
            )
        elif command == "/new":
            if self.run_id and self._status not in FINAL_STATUSES:
                self._chat(
                    "Atlant",
                    "Запуск ещё работает. Остановить: /stop. Закрыть UI без остановки: /quit.",
                )
                return
            self.run_id = None
            self.event_seq = 0
            self._snapshot = {}
            self._reported_result = None
            self._render_snapshot({"status": "idle"})
            self.query_one("#chat", RichLog).clear()
            self._welcome()
        elif command == "/mode":
            if arg not in {"agent", "generate"}:
                self._chat(
                    "Режим",
                    "/mode agent — работа с проектом\n/mode generate — генерация",
                )
            elif self.run_id and self._status not in FINAL_STATUSES:
                self._chat(
                    "Atlant",
                    "Сменить режим можно после завершения или остановки запуска.",
                )
            else:
                self._mode = arg
                self._render_status()
        elif command == "/runs":
            runs = await self._guarded("runs")
            if runs is not None:
                self._runs = runs
                self._chat(
                    "Запуски",
                    (
                        "\n".join(
                            f"{i}. {STATUS.get(r['status'], r['status'])} · {' '.join(r.get('task', '').split())[:90]}"
                            for i, r in enumerate(runs, 1)
                        )
                        + "\n\nПодключиться: /attach номер или ID"
                        if runs
                        else "Пока нет запусков."
                    ),
                )
        elif command == "/attach":
            if arg.isdecimal() and 1 <= int(arg) <= len(self._runs):
                arg = self._runs[int(arg) - 1]["id"]
            if arg:
                await self._attach(arg)
            else:
                self._chat("Atlant", "Укажи ID запуска или выбери номер из /runs.")
        elif command in {"/stop", "/resume"}:
            if not self.run_id:
                self._chat(
                    "Atlant", "Нет подключённого запуска. Прошлые запуски: /runs."
                )
                return
            if command == "/resume" and not self._snapshot.get(
                "can_resume", self._status in {"stopped", "interrupted"}
            ):
                self._chat(
                    "Atlant",
                    "Этот запуск нельзя продолжить. Можно отправить новую задачу.",
                )
                return
            if await self._guarded(command[1:], self.run_id) is not None:
                if command == "/resume":
                    self._reported_result = None
                snapshot = await self._guarded("snapshot", self.run_id)
                if snapshot:
                    self._render_snapshot(snapshot)
        elif command in {"/allow", "/deny"}:
            if not self._approval_id or not self.run_id:
                self._chat("Atlant", "Сейчас нет запроса разрешения.")
            else:
                await self._respond_approval(command == "/allow")
        elif command in {"/diff", "/checks", "/agents", "/usage"}:
            self._details(command)
        else:
            self._chat("Atlant", "Неизвестная команда. Список: /help.")

    def _details(self, command: str) -> None:
        state = self._snapshot
        if command == "/diff":
            rows = [
                f"{c.get('path', '')}\n{c.get('diff', '')}"
                for c in state.get("changes", [])
            ]
        elif command == "/checks":
            rows = [
                f"{c.get('status', 'UNKNOWN')} · {c.get('check', '')}\n{c.get('detail', '')}".rstrip()
                for c in state.get("checks", [])
            ]
        elif command == "/agents":
            rows = [
                (
                    " · ".join(
                        str(a[k])
                        for k in ("role", "status", "file", "task")
                        if a.get(k)
                    )
                    if isinstance(a, dict)
                    else str(a)
                )
                for a in state.get("agents", [])
            ]
        else:
            rows = [
                f"{USAGE_LABELS.get(k, k)}: {'неизвестно' if v is None else v}"
                for k, v in {
                    **state.get("usage", {}),
                    **state.get("counters", {}),
                }.items()
            ]
        self._chat(
            {
                "/diff": "Изменения",
                "/checks": "Проверки",
                "/agents": "Агенты",
                "/usage": "Использование",
            }[command],
            "\n\n".join(rows) or "Пока нет данных.",
        )

    def action_runs(self) -> None:
        self.run_worker(self._command("/runs"))

    def action_stop_run(self) -> None:
        if not self._starting and (not self.run_id or self._status in FINAL_STATUSES):
            return
        self._stop_requested = True
        self._render_snapshot({**self._snapshot, "status": "stopping"})
        if not self._starting:
            self.run_worker(self._request_stop(), group="stop")

    async def _request_stop(self) -> None:
        if not self.run_id or self._stop_inflight:
            return
        self._stop_inflight = True
        run_id = self.run_id
        try:
            result = await self._guarded("stop", run_id, control=True)
            if result is not None and "status" not in result:
                result = await self._guarded("snapshot", run_id, control=True)
            if result and run_id == self.run_id:
                self._render_snapshot(result)
        finally:
            self._stop_inflight = False

    def action_cycle_mode(self) -> None:
        if self._busy or (self.run_id and self._status not in FINAL_STATUSES):
            return
        self._mode = "generate" if self._mode == "agent" else "agent"
        self._render_status()

    def on_resize(self) -> None:
        if self.query("#status"):
            self.call_after_refresh(self._render_status)

    def action_history_page(self, direction: int) -> None:
        chat = self.query_one("#chat", RichLog)
        if direction < 0:
            chat.scroll_page_up(animate=False)
        else:
            chat.scroll_page_down(animate=False)

    def action_focus_prompt(self) -> None:
        self.query_one("#task", Prompt).focus()

    def action_focus_approval(self) -> None:
        approval = self.query_one("#approval", OptionList)
        if self._approval_id and not approval.has_focus:
            approval.focus()
        else:
            self.action_focus_prompt()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "approval":
            self.run_worker(self._respond_approval(event.option_index == 0))

    async def _respond_approval(self, allow: bool) -> None:
        if not self.run_id or not self._approval_id or self._approval_inflight:
            return
        self._approval_inflight = True
        run_id, approval_id = self.run_id, self._approval_id
        try:
            result = await self._guarded(
                "approve", run_id, approval_id, allow, control=True
            )
            if result is not None:
                snapshot = await self._guarded("snapshot", run_id, control=True)
                if snapshot and run_id == self.run_id:
                    self._render_snapshot(snapshot)
        finally:
            self._approval_inflight = False
            if not self._approval_id and self.query("#task"):
                self.action_focus_prompt()

    async def _attach(self, run_id: str) -> None:
        snapshot = await self._guarded("snapshot", run_id)
        if snapshot is None:
            return
        self.run_id = run_id
        self._mode = snapshot.get("mode", self._mode)
        self.event_seq = 0
        self._local_messages.clear()
        self._last_activity = ""
        self._reported_result = None
        self.query_one("#chat", RichLog).clear()
        events = await self._guarded("events", run_id, 0)
        if events:
            self._events(events)
        self._render_snapshot(snapshot)

    def poll_run(self) -> None:
        if not self._polling and not self._busy:
            self.run_worker(self.refresh_run(), group="poll")

    async def refresh_run(self) -> None:
        if not self.run_id or self._busy or self._polling:
            return
        self._polling = True
        run_id = self.run_id
        try:
            events, snapshot = await asyncio.gather(
                self._guarded("events", run_id, self.event_seq),
                self._guarded("snapshot", run_id),
            )
            if run_id != self.run_id:
                return
            if events:
                self._events(events)
            if snapshot:
                self._render_snapshot(snapshot)
        finally:
            self._polling = False

    def _events(self, events: list[dict]) -> None:
        if not self.query("#chat"):
            return
        for event in events:
            if int(event.get("seq", 0)) <= self.event_seq:
                continue
            self.event_seq = max(self.event_seq, int(event.get("seq", 0)))
            kind = event.get("kind")
            message = event.get("message", "")
            if kind in {"task", "user"} and message in self._local_messages:
                self._local_messages.remove(message)
                continue
            if kind in {"stage", "tool", "check"}:
                activity = ""
                if kind == "tool":
                    data = event.get("data") or {}
                    action = {
                        "READ": "Читаю",
                        "LIST": "Смотрю файлы",
                        "SEARCH": "Ищу",
                        "RUN": "Выполняю проверку",
                    }.get(data.get("command"), "Изучаю проект")
                    activity = f"{action} {data.get('path', '')}".rstrip()
                elif kind == "check":
                    activity = "Проверяю проект"
                elif message.startswith("file_completed:"):
                    activity = "Изменён " + message.partition(":")[2]
                elif message in {
                    "planning",
                    "coordinator_started",
                    "contract_approved",
                    "post_test_repair",
                }:
                    activity = STAGES[message]
                if activity and activity != self._last_activity:
                    self._last_activity = activity
                continue
            if kind in {"result", "approval", "error", "interrupted", "stopped"}:
                continue
            self._chat("Вы" if kind in {"task", "user"} else "Atlant", message)

    def _render_snapshot(self, snapshot: dict[str, Any]) -> None:
        if not self.query("#approval"):
            return
        self._snapshot = snapshot
        self._status = str(snapshot.get("status", "idle"))
        # A finished run's mode must not overwrite the choice for the next task.
        if self._status not in FINAL_STATUSES:
            self._mode = snapshot.get("mode", self._mode)
        self._render_status()
        self._render_plan()
        label = STATUS.get(self._status, self._status)
        approval = snapshot.get("pending_approval")
        if approval and isinstance(approval.get("allow"), bool):
            approval = None
        approval_id = (
            str(approval.get("id") or approval.get("approval_id")) if approval else None
        )
        if approval_id and approval_id != self._approval_id:
            self._chat(
                "Нужно разрешение",
                str(
                    approval.get("description")
                    or approval.get("message")
                    or "Разрешить действие?"
                )
                + "\n/allow — разрешить · /deny — отклонить",
            )
        new_approval = approval_id and approval_id != self._approval_id
        self._approval_id = approval_id
        selector = self.query_one("#approval", OptionList)
        selector.set_class(bool(approval), "visible")
        if new_approval:
            # An Enter already queued while typing must never authorize a command.
            selector.highlighted = 1
            selector.focus()
        elif not approval and selector.has_focus:
            self.action_focus_prompt()
        self._render_status()
        result_key = (self.run_id, self._status)
        if self._status in FINAL_STATUSES and self._reported_result != result_key:
            self._reported_result = result_key
            changes, checks = snapshot.get("changes", []), snapshot.get("checks", [])
            paths = [str(c.get("path", "")).lstrip("/") for c in changes]
            lines = ["Изменено: " + (", ".join(paths[:5]) or "нет файлов")]
            if len(paths) > 5:
                lines[-1] += f" и ещё {len(paths) - 5} (/diff)"
            failed = sum(c.get("status") == "FAIL" for c in checks)
            passed = sum(c.get("status") == "PASS" for c in checks)
            unknown = len(checks) - passed - failed
            if failed:
                lines.append(
                    f"Проверка: ошибки в {failed} проверках. Подробнее: /checks."
                )
            elif passed and not unknown:
                lines.append(f"Проверка: пройдена ({passed}).")
            else:
                lines.append(
                    "Проверка: результат не подтверждён. " + self._check_reason()
                )
            summary = (
                snapshot.get("result_summary")
                if self._status in {"completed", "unverified"}
                else None
            )
            lines.append("Результат: " + (summary or self._result_reason()))
            if self._status in {"failed", "stopped", "interrupted"}:
                lines.append(
                    (
                        "Продолжить: /resume. "
                        if snapshot.get("can_resume")
                        else (
                            "Повторить: /retry. "
                            if snapshot.get("task")
                            else "Повторить: отправь задачу ещё раз. "
                        )
                    )
                    + "Подробнее: /details."
                )
            self._chat(label, "\n".join(lines))

    def _check_reason(self) -> str:
        checks = self._snapshot.get("checks", [])
        for check in checks:
            if check.get("status") not in {"PASS", "FAIL"}:
                reason = check.get("reason") or check.get("message")
                if reason:
                    return str(reason).splitlines()[0][:180]
        if self._status in {"stopped", "interrupted", "failed"}:
            return "Запуск завершился до подтверждения результата."
        return "Автоматические проверки проекта не настроены или не завершились. Подробнее: /checks."

    def _result_reason(self) -> str:
        state = self._snapshot
        if self._status == "stopped":
            return "Остановлено пользователем. Уже применённые изменения сохранены."
        if self._status in {"interrupted", "failed"}:
            reason = state.get("interruption_reason") or state.get("error")
            return str(reason) if reason else "Причина прерывания неизвестна."
        if self._status == "completed":
            return "План выполнен, настроенные проверки пройдены."
        if self._status == "unverified":
            return "Изменения сохранены; работоспособность ещё не подтверждена."
        return "Запуск ещё не завершён."

    def _plan_text(self, limit: int | None = None) -> Text:
        plan = self._snapshot.get("plan", [])
        text = Text()
        start = 0
        if limit:
            active = next(
                (i for i, step in enumerate(plan) if step.get("status") == "running"), 0
            )
            start = max(0, active - 1)
        shown = plan[start : start + limit] if limit else plan
        for i, step in enumerate(shown):
            if i:
                text.append("\n")
            status = step.get("status", "pending")
            marker = {"completed": "✓", "running": "›"}.get(status, "·")
            style = "bold " + COLORS["mode"] if status == "running" else COLORS["muted"]
            description = " ".join(
                str(step.get("description") or step.get("path", "")).split()
            )
            line = Text(f"{marker} {description}", style=style)
            if limit:
                line.truncate(max(1, self.size.width - 2), overflow="ellipsis")
            text.append_text(line)
        if limit and len(plan) > limit:
            text.append(f"\nВсе {len(plan)} шагов: /plan", style=COLORS["subtle"])
        return text

    def _render_plan(self) -> None:
        widget = self.query_one("#plan", Static)
        visible = (
            bool(self._snapshot.get("plan")) and self._status not in FINAL_STATUSES
        )
        widget.set_class(visible, "visible")
        if visible:
            preliminary = self._snapshot.get("stage") in {
                "plan_ready",
                "planning",
                "coordinator_started",
                "candidate_plan_ready",
            }
            content = Text()
            content.append(
                "Предварительный план\n" if preliminary else "План\n", style="bold"
            )
            content.append_text(self._plan_text(4))
            widget.update(content)

    def _render_status(self) -> None:
        if not self.query("#status"):
            return
        snapshot = self._snapshot
        active = bool(
            self._starting or (self.run_id and self._status not in FINAL_STATUSES)
        )
        stage = str(snapshot.get("stage", ""))
        label = STATUS.get(self._status, self._status)
        action = ""
        if self._status == "running":
            label = (
                "Думает"
                if stage
                in {
                    "planning",
                    "plan_ready",
                    "coordinator_started",
                    "candidate_plan_ready",
                    "replan_requested",
                }
                else "Выполняет"
            )
            agents = snapshot.get("agents", [])
            current = next(
                (
                    a.get("file")
                    for a in agents
                    if a.get("status") == "running" and a.get("file")
                ),
                None,
            )
            action = (
                ("Изменяет " + str(current).lstrip("/"))
                if current
                else self._last_activity or STAGES.get(stage, "Готовит следующий шаг")
            )
        key = (self.run_id, self._status, stage, action)
        if key != self._activity_key:
            self._activity_key, self._activity_since = key, time.monotonic()
        activity = Text(
            label,
            style="bold "
            + COLORS[
                (
                    "warning"
                    if self._status
                    in {"failed", "interrupted", "stopped", "waiting_approval"}
                    else "mode" if active else "text"
                )
            ],
        )
        if action:
            activity.append(" · " + action, style=COLORS["muted"])
        if active:
            activity.append(
                f" · {int(time.monotonic() - self._activity_since)} с",
                style=COLORS["subtle"],
            )
        activity.truncate(max(1, self.size.width - 2), overflow="ellipsis")
        self.query_one("#activity", Static).update(activity)

        model = snapshot.get("model") or os.environ.get(
            "CODEX_GATEWAY_MODEL", "gpt-5.6-sol"
        )
        width = max(1, self.size.width - 2)
        mode = "Agent" if self._mode == "agent" else "Generate"
        usage = snapshot.get("usage", {})
        context = usage.get("context_percent")
        tokens = [usage.get(k) for k in ("input_tokens", "output_tokens")]
        metrics = ""
        if isinstance(context, (int, float)) and 0 <= context <= 100:
            metrics = f" · контекст {context:.0f}%"
        elif all(type(n) is int and n >= 0 for n in tokens):
            total = sum(tokens)
            count = (
                f"{total / 1_000_000:.1f}m"
                if total >= 1_000_000
                else f"{total / 1000:.1f}k" if total >= 1000 else str(total)
            )
            metrics = f" · {count} tok"
        tail = f" · {model} · {mode}" + metrics
        folder = Text(Path(self.workspace).name or self.workspace)
        folder.truncate(max(1, width - len(tail)), overflow="ellipsis")
        top = Text(folder.plain, style=COLORS["muted"])
        top.append(f" · {model}", style=COLORS["model"])
        top.append(f" · {mode}", style=COLORS["muted"])
        top.append(metrics, style=COLORS["identity"])
        top.truncate(width, overflow="ellipsis")
        self.query_one("#status", Static).update(top)
        hint = "Enter отправить · Alt+Enter перенос · Esc остановить · Shift+Tab режим"
        if self._approval_id:
            hint = "↑↓ выбор · Enter подтвердить · Tab ввод · Esc остановить"
        self.query_one("#hint", Static).update(
            Text(hint, style=COLORS["subtle"], no_wrap=True, overflow="ellipsis")
        )
