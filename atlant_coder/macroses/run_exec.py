import os
import tempfile
from pathlib import Path
from typing import Union

# официальный ответ разработчика об этом: Это чтобы наверняка


def write_file_after_generic(file_path: Union[str, os.PathLike], code: str) -> None:
    """
    Безопасная и атомарная запись файла с созданием всех промежуточных директорий.
    Поддерживает .dot-папки и гарантирует, что файл не будет создан как директория.

    Args:
        file_path (str | PathLike): Абсолютный или относительный путь к файлу.
        code (str): Содержимое файла.

    Raises:
        ValueError: Если путь ведёт к существующей директории.
        PermissionError: Если нет прав на создание директорий или запись.
        OSError: Если произошла системная ошибка при записи.
    """
    path = Path(file_path).resolve(strict=False)
    previous_mode = path.stat().st_mode & 0o777 if path.is_file() else None

    # Безопасность: не позволим случайно перезаписать директорию
    if path.exists() and path.is_dir():
        raise ValueError(f"Ожидался путь к файлу, но получена директория: {path}")

    # Создаём директории до файла, исключая сам файл (включая .dot-папки)
    try:
        current = Path()
        for part in path.parts[:-1]:
            current = current / part
            if current.exists():
                if current.is_file():
                    raise ValueError(
                        f"Часть пути {current} — файл, нельзя создать директорию."
                    )
                continue
            current.mkdir(mode=0o755)

        # Проверка прав на запись
        if not os.access(path.parent, os.W_OK):
            raise PermissionError(f"Нет прав на запись в директорию: {path.parent}")

        # Создание временного файла и атомарная замена
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=path.parent
        ) as tmp:
            if previous_mode is not None:
                os.fchmod(tmp.fileno(), previous_mode)
            tmp.write(code)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_path = Path(tmp.name)

        os.replace(temp_path, path)

    except Exception as e:
        raise OSError(f"Ошибка при записи в файл '{file_path}': {e}")
