def generate_list_from_structure(structure) -> list:
    """Return a flat list of file paths from an ARCH-produced structure.

    Handles three shapes:
      1. Flat ``{path: desc}`` (current ARCH schema) → list of keys.
      2. Flat ``{path: {desc, priority, ...}}`` (extended schema) → same.
      3. Legacy dir-nested ``{dir: [file1, file2]}`` → joined ``dir/file`` paths.
      4. Already a list/tuple → returned as-is.

    Empty or non-iterable input returns an empty list.
    """
    if not structure:
        return []

    if isinstance(structure, (list, tuple)):
        return list(structure)

    if not isinstance(structure, dict):
        return []

    def _looks_like_dir_nested(d: dict) -> bool:
        for value in d.values():
            if isinstance(value, list):
                return all(isinstance(item, str) for item in value)
            return False
        return False

    if _looks_like_dir_nested(structure):
        result: list = []
        for dir_path, files in structure.items():
            prefix = dir_path if dir_path.endswith("/") else f"{dir_path}/"
            for name in files or []:
                result.append(f"{prefix}{name}")
        return result

    return list(structure.keys())
