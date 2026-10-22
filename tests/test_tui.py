from __future__ import annotations

import asyncio
import threading

import pytest
from textual.widgets import OptionList, RichLog

from atlant_coder.tui import AtlantApp, Prompt


class FakeClient:
    def __init__(self):
        self.calls = []
        self.state = {
            "id": "run-1",
            "status": "running",
            "stage": "generating",
            "mode": "agent",
            "agents": [{"role": "CODER", "file": "/app.py", "status": "running"}],
            "changes": [{"path": "app.py", "diff": "+print('ok')"}],
            "checks": [{"check": "pytest", "status": "PASS"}],
            "usage": {"input_tokens": None, "output_tokens": 42},
            "counters": {"architecture_reworks": 0, "post_test_repairs": 2},
        }

    def start(self, workspace, task, mode="agent"):
        self.calls.append(("start", workspace, task, mode))
        self.state.update(status="running", mode=mode)
        return {"id": "run-1"}

    def snapshot(self, run_id):
        return dict(self.state)

    def events(self, run_id, after=0):
        return (
            []
            if after
            else [
                {"seq": 1, "kind": "task", "message": "Собери проект", "data": {}},
                {
                    "seq": 2,
                    "kind": "stage",
                    "message": "coordinator_started",
                    "data": {},
                },
                {"seq": 3, "kind": "tool", "message": "READ", "data": {}},
            ]
        )

    def steer(self, run_id, message):
        self.calls.append(("steer", run_id, message))
        return {"ok": True}

    def stop(self, run_id):
        self.calls.append(("stop", run_id))
        self.state.update(status="stopped", can_resume=True)
        return {"ok": True}

    def resume(self, run_id):
        self.calls.append(("resume", run_id))
        self.state["status"] = "running"
        return {"ok": True}

    def approve(self, run_id, approval_id, allow):
        self.calls.append(("approve", run_id, approval_id, allow))
        self.state.pop("pending_approval", None)
        return {"ok": True}

    def runs(self):
        return [
            {"id": "run-1", "status": self.state["status"], "task": "Собери проект"}
        ]


def chat(app):
    return "\n".join(line.text for line in app.query_one("#chat", RichLog).lines)


async def submit(app, pilot, text):
    app.query_one("#task", Prompt).load_text(text)
    app.query_one("#task", Prompt).focus()
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
def test_chat_first_layout_and_keyboard_input(size):
    async def scenario():
        client = FakeClient()
        app = AtlantApp(client, "/work")
        async with app.run_test(size=size) as pilot:
            assert not app.query("#side, #toolbar, #actions")
            assert not app.query_one("#approval").display
            assert app.query_one("#chat").content_region.height >= size[1] - 10
            prompt = app.query_one("#task", Prompt)
            assert prompt.has_focus
            await pilot.press("a", "alt+enter", "b")
            assert prompt.text == "a\nb"
            await pilot.press("enter")
            await pilot.pause()
            assert client.calls == [("start", "/work", "a\nb", "agent")]
            assert prompt.text == ""
            assert prompt.region.bottom < size[1]
            await submit(app, pilot, "Уточнение")
            assert ("steer", "run-1", "Уточнение") in client.calls
            app.save_screenshot(f"atlant-chat-{size[0]}.svg", path="/tmp")

    asyncio.run(scenario())


def test_details_are_on_demand_and_internal_events_stay_out_of_chat():
    async def scenario():
        app = AtlantApp(FakeClient(), "/work", "run-1")
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert "Собери проект" in chat(app)
            assert "coordinator_started" not in chat(app)
            assert "READ" not in chat(app)
            assert "input_tokens" not in chat(app)
            for command in ("/diff", "/checks", "/agents", "/usage"):
                await submit(app, pilot, command)
            text = chat(app)
            assert "+print('ok')" in text
            assert "PASS · pytest" in text
            assert "CODER · running · /app.py" in text
            assert "Входные токены: неизвестно" in text
            assert "Переработки архитектуры: 0" in text
            assert "Исправления после тестов: 2" in text
            app._chat("Вы", "Очень длинная строка " * 30)
            await pilot.pause()
            assert app.query_one("#chat").max_scroll_x == 0

    asyncio.run(scenario())


def test_runs_attach_stop_resume_and_new_conversation():
    async def scenario():
        client = FakeClient()
        app = AtlantApp(client, "/work")
        async with app.run_test() as pilot:
            await submit(app, pilot, "/runs")
            assert "1. Выполняет" in chat(app)
            await submit(app, pilot, "/attach 1")
            assert app.run_id == "run-1"
            await submit(app, pilot, "/new")
            assert app.run_id == "run-1"  # Do not abandon an active writer silently.
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert ("stop", "run-1") in client.calls
            await submit(app, pilot, "/resume")
            assert ("resume", "run-1") in client.calls
            await submit(app, pilot, "/stop")
            await submit(app, pilot, "/new")
            assert app.run_id is None
            assert "Собери проект" not in chat(app)
            await submit(app, pilot, "/mode generate")
            await submit(app, pilot, "Новая игра")
            assert client.calls[-1] == ("start", "/work", "Новая игра", "generate")

    asyncio.run(scenario())


def test_approval_is_contextual_and_denial_reaches_service():
    async def scenario():
        client = FakeClient()
        client.state.update(
            status="waiting_approval",
            pending_approval={
                "id": "approval-1",
                "description": "Запустить внешнюю команду?",
            },
        )
        app = AtlantApp(client, "/work", "run-1")
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert app.query_one("#approval").display
            assert "Запустить внешнюю команду?" in chat(app)
            assert not app.query("Button")
            selector = app.query_one("#approval", OptionList)
            assert selector.has_focus
            assert selector.highlighted == 1
            await pilot.press("enter")
            await pilot.pause()
            assert ("approve", "run-1", "approval-1", False) in client.calls
            assert not app.query_one("#approval").display
            client.state["pending_approval"] = {
                "id": "approval-2",
                "description": "Вторая команда",
            }
            await app.refresh_run()
            await pilot.pause()
            assert selector.has_focus
            await pilot.press("up", "enter")
            await pilot.pause()
            assert ("approve", "run-1", "approval-2", True) in client.calls
            assert app.query_one("#task", Prompt).has_focus

    asyncio.run(scenario())


def test_completed_result_is_reported_once_and_followup_starts_new_run():
    async def scenario():
        client = FakeClient()
        client.state.update(status="completed", mode="generate")
        app = AtlantApp(client, "/work", "run-1")
        async with app.run_test() as pilot:
            await pilot.pause()
            before = chat(app)
            assert "Готово" in before
            assert "Проверка: пройдена (1)." in before
            await app.refresh_run()
            await pilot.pause()
            assert chat(app) == before
            await submit(app, pilot, "Добавь поиск")
            assert client.calls[-1] == ("start", "/work", "Добавь поиск", "generate")

    asyncio.run(scenario())


def test_slow_start_keeps_typing_and_detach_responsive():
    class SlowClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.started, self.release = threading.Event(), threading.Event()

        def start(self, *args):
            self.started.set()
            self.release.wait(timeout=5)
            return super().start(*args)

    async def scenario():
        client = SlowClient()
        app = AtlantApp(client, "/work")
        async with app.run_test(size=(80, 24)) as pilot:
            prompt = app.query_one("#task", Prompt)
            prompt.load_text("Долгая задача")
            await pilot.press("enter")
            assert await asyncio.to_thread(client.started.wait, 1)
            await asyncio.wait_for(pilot.press("x"), timeout=0.5)
            assert prompt.text == "x"
            try:
                await asyncio.wait_for(pilot.press("ctrl+q"), timeout=0.5)
                assert not any(c[0] == "stop" for c in client.calls)
            finally:
                client.release.set()

    asyncio.run(scenario())


@pytest.mark.parametrize("key", ["escape", "ctrl+c", "ctrl+s"])
def test_stop_shortcuts_work_while_editor_has_focus(key):
    async def scenario():
        client = FakeClient()
        app = AtlantApp(client, "/work", "run-1")
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one("#task", Prompt)
            prompt.load_text("Не терять мой черновик")
            prompt.focus()
            await pilot.press(key)
            await pilot.pause()
            assert ("stop", "run-1") in client.calls
            assert prompt.text == "Не терять мой черновик"
            assert app._status == "stopped"

    asyncio.run(scenario())


def test_stop_before_start_response_is_remembered():
    class SlowStart(FakeClient):
        def __init__(self):
            super().__init__()
            self.entered, self.release = threading.Event(), threading.Event()

        def start(self, *args):
            self.entered.set()
            self.release.wait(5)
            return super().start(*args)

    async def scenario():
        client = SlowStart()
        app = AtlantApp(client, "/work")
        async with app.run_test() as pilot:
            prompt = app.query_one("#task", Prompt)
            prompt.load_text("Задача")
            await pilot.press("enter")
            assert await asyncio.to_thread(client.entered.wait, 1)
            try:
                prompt.load_text("/stop")
                await pilot.press("enter")
                await pilot.pause()
                assert app._stop_requested
                assert app._status == "stopping"
            finally:
                client.release.set()
            await app.workers.wait_for_complete()
            assert ("stop", "run-1") in client.calls
            assert app._status == "stopped"

    asyncio.run(scenario())


def test_stop_bypasses_busy_send_and_poll_threads():
    class BlockedClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.block = False
            self.poll_entered, self.send_entered = threading.Event(), threading.Event()
            self.stopped, self.release = threading.Event(), threading.Event()

        def snapshot(self, run_id):
            if self.block:
                self.poll_entered.set()
                self.release.wait(5)
            return super().snapshot(run_id)

        def steer(self, *args):
            self.send_entered.set()
            self.release.wait(5)
            return super().steer(*args)

        def stop(self, run_id):
            super().stop(run_id)
            self.stopped.set()
            return dict(self.state)

    async def scenario():
        client = BlockedClient()
        app = AtlantApp(client, "/work", "run-1")
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            client.block = True
            poll = asyncio.create_task(app.refresh_run())
            assert await asyncio.to_thread(client.poll_entered.wait, 1)
            app.query_one("#task", Prompt).load_text("Уточнение")
            await pilot.press("enter")
            assert await asyncio.to_thread(client.send_entered.wait, 1)
            try:
                await pilot.press("escape")
                assert await asyncio.to_thread(client.stopped.wait, 0.5)
                await pilot.pause()
                assert app._status == "stopped"
            finally:
                client.release.set()
                await poll

    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
def test_status_bar_real_metrics_modes_and_resize(size):
    async def scenario():
        client = FakeClient()
        app = AtlantApp(
            client, "/very/long/workspace/path/with/many/directories/project"
        )
        async with app.run_test(size=size) as pilot:
            app._render_snapshot(
                {
                    "status": "idle",
                    "usage": {
                        "input_tokens": 200000,
                        "output_tokens": 54500,
                        "cached_input_tokens": 100000,
                    },
                }
            )
            top = app.query_one("#status").render()
            assert "254.5k tok" in top.plain
            assert "cache" not in top.plain
            assert top.cell_length <= size[0] - 2
            assert len({s.style for s in top.spans}) >= 3
            await pilot.press("shift+tab")
            assert app._mode == "generate"
            assert "Generate" in app.query_one("#status").render().plain
            app._render_snapshot(
                {
                    "status": "running",
                    "usage": {"input_tokens": None, "output_tokens": 2},
                    "agents": [{"role": "CODER", "status": "running"}],
                }
            )
            app.run_id = "run-1"
            app._render_status()
            assert "tok" not in app.query_one("#status").render().plain
            assert "cache" not in app.query_one("#status").render().plain
            assert "Esc остановить" in app.query_one("#hint").render().plain
            assert "agents:" not in app.query_one("#hint").render().plain
            await pilot.press("shift+tab")
            assert app._mode == "generate"  # Existing run mode cannot change silently.
            await pilot.resize_terminal(80, 24)
            assert app.query_one("#status").render().cell_length <= 78

    asyncio.run(scenario())


def test_actual_plan_one_state_and_human_result():
    async def scenario():
        client = FakeClient()
        client.state.update(
            plan=[
                {
                    "path": "index.html",
                    "description": "Создать интерфейс",
                    "status": "completed",
                },
                {
                    "path": "app.js",
                    "description": "Добавить вычисления",
                    "status": "running",
                },
                {
                    "path": "test.js",
                    "description": "Проверить ввод",
                    "status": "pending",
                },
            ],
        )
        app = AtlantApp(client, "/work", "run-1")
        async with app.run_test(size=(80, 24)) as pilot:
            await app.workers.wait_for_complete()
            plan = app.query_one("#plan").render()
            assert "✓ Создать интерфейс" in plan.plain
            assert "› Добавить вычисления" in plan.plain
            assert "· Проверить ввод" in plan.plain
            assert app.query_one("#plan").display
            assert "Выполняет" in app.query_one("#activity").render().plain
            assert "Изменяет app.py" in app.query_one("#activity").render().plain
            assert "0 с" in app.query_one("#activity").render().plain
            assert "Изучаю проект" not in chat(app)
            await submit(app, pilot, "/plan")
            assert "Добавить вычисления" in chat(app)
            client.state.update(
                status="interrupted",
                mode="generate",
                can_resume=True,
                interruption_reason="Сервис не передал идентификатор попытки.",
                result_summary="Не показывать старый успешный итог",
                checks=[],
            )
            await app.refresh_run()
            await pilot.pause()
            assert app.query_one("#activity").render().plain == "Прерван"
            assert not app.query_one("#plan").display
            result = chat(app)
            assert "Сервис не передал идентификатор попытки." in result
            assert "Не показывать старый успешный итог" not in result
            assert "UNKNOWN" not in result
            assert "Продолжить: /resume" in result
            assert "Подробнее: /details" in result
            app._events([{"seq": 9, "kind": "error", "message": "raw traceback"}])
            await app.refresh_run()
            await pilot.pause()
            assert chat(app) == result
            await submit(app, pilot, "/details")
            assert chat(app).count("Сервис не передал идентификатор попытки.") == 2

    asyncio.run(scenario())


def test_plan_keeps_active_step_visible_after_many_completed_steps():
    async def scenario():
        app = AtlantApp(FakeClient(), "/work", "run-1")
        async with app.run_test(size=(80, 24)) as pilot:
            await app.workers.wait_for_complete()
            plan = [
                {
                    "description": f"Шаг {i}",
                    "status": "completed" if i < 5 else "running",
                }
                for i in range(6)
            ]
            app._render_snapshot({"status": "running", "plan": plan})
            assert "› Шаг 5" in app.query_one("#plan").render().plain
            app._render_snapshot({"status": "interrupted"})
            await pilot.pause()
            assert "Причина прерывания неизвестна." in chat(app)
            assert "Повторить:" in chat(app)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status", ["completed", "failed", "interrupted", "stopped", "unverified"]
)
@pytest.mark.parametrize("switch", ["shift+tab", "/mode generate"])
def test_next_run_mode_survives_poll_of_finished_run(status, switch):
    async def scenario():
        client = FakeClient()
        client.state.update(status=status, mode="agent")
        app = AtlantApp(client, "/work", "run-1")
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            if switch.startswith("/"):
                await submit(app, pilot, switch)
            else:
                await pilot.press(switch)
            assert app._mode == "generate"
            for _ in range(2):
                await app.refresh_run()
            assert app._mode == "generate"
            assert "Generate" in app.query_one("#status").render().plain
            await submit(app, pilot, "Следующая задача")
            assert ("start", "/work", "Следующая задача", "generate") in client.calls

    asyncio.run(scenario())


def test_keyboard_history_and_approval_preserve_draft_and_selection():
    async def scenario():
        client = FakeClient()
        app = AtlantApp(client, "/work", "run-1")
        async with app.run_test(size=(80, 24)) as pilot:
            await app.workers.wait_for_complete()
            prompt = app.query_one("#task", Prompt)
            prompt.load_text("Несохранённое уточнение")
            for i in range(40):
                app._chat("Atlant", f"Строка {i}")
            await pilot.pause()
            log = app.query_one("#chat", RichLog)
            bottom = log.scroll_y
            await pilot.press("pageup")
            assert log.scroll_y < bottom
            await pilot.press("pagedown")
            assert log.scroll_y == bottom
            assert prompt.has_focus
            client.state["pending_approval"] = {
                "id": "p1",
                "description": "Проверить проект?",
            }
            await app.refresh_run()
            await pilot.pause()
            selector = app.query_one("#approval", OptionList)
            assert selector.has_focus
            assert "↑↓ выбор" in app.query_one("#hint").render().plain
            await pilot.press("up")
            await app.refresh_run()
            assert selector.highlighted == 0
            await pilot.press("tab")
            assert prompt.has_focus
            assert prompt.text == "Несохранённое уточнение"
            await pilot.press("tab")
            assert selector.has_focus
            await pilot.press("enter")
            await pilot.pause()
            assert ("approve", "run-1", "p1", True) in client.calls
            assert prompt.has_focus
            assert prompt.text == "Несохранённое уточнение"
            assert not any(c[0] == "steer" for c in client.calls)
            client.state["pending_approval"] = {
                "id": "p2",
                "description": "Вторая проверка",
            }
            await app.refresh_run()
            await pilot.press("ctrl+l")
            assert prompt.has_focus
            await submit(app, pilot, "/deny")
            assert ("approve", "run-1", "p2", False) in client.calls

    asyncio.run(scenario())
