import os


def load_env_file(env_path='.env', override=False):
    if not env_path or not os.path.exists(env_path):
        return

    with open(env_path, 'r', encoding='utf-8') as env_file:
        for raw_line in env_file:
            line = raw_line.strip()

            if not line or line.startswith('#') or '=' not in line:
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if not key:
                continue

            if override or key not in os.environ:
                os.environ[key] = value


def _smb_to_unc(path):
    if not isinstance(path, str):
        return path

    raw_path = path.strip()
    if not raw_path.lower().startswith("smb://"):
        return raw_path

    # Convert smb://server/share/folder to \\server\share\folder for Windows.
    parts = raw_path[6:].lstrip("/").split("/")
    if len(parts) < 2:
        return raw_path

    server = parts[0]
    share = parts[1]
    remainder = parts[2:]

    unc_path = "\\\\" + server + "\\" + share
    if remainder:
        unc_path = unc_path + "\\" + "\\".join(remainder)
    return unc_path


def resolve_path(path_value, nas_root=None, base_dir=None):
    if path_value is None:
        return None

    path = str(path_value).strip()
    if not path:
        return path

    path = _smb_to_unc(path)
    path = os.path.expandvars(os.path.expanduser(path))

    normalized_nas_root = None
    if nas_root:
        normalized_nas_root = str(nas_root).strip()
        normalized_nas_root = _smb_to_unc(normalized_nas_root)
        normalized_nas_root = os.path.expandvars(os.path.expanduser(normalized_nas_root))

    if not os.path.isabs(path):
        if normalized_nas_root:
            path = os.path.join(normalized_nas_root, path)
        elif base_dir:
            path = os.path.join(base_dir, path)

    return os.path.normpath(path)


def resolve_conf_paths(conf, keys, base_dir=None, nas_root_key="nas_root"):
    env_base_dir = base_dir if base_dir else os.getcwd()
    load_env_file(os.path.join(env_base_dir, '.env'))

    nas_root = getattr(conf, nas_root_key, None)
    if not nas_root:
        nas_root = os.environ.get('SASHA_NAS_ROOT')

    resolved_paths = {}

    for key in keys:
        if not hasattr(conf, key):
            continue

        value = getattr(conf, key)
        if value is None or str(value).strip() == "":
            continue

        resolved_value = resolve_path(value, nas_root=nas_root, base_dir=base_dir)
        setattr(conf, key, resolved_value)
        resolved_paths[key] = resolved_value

    return resolved_paths


def ensure_path_exists(path, field_name, expect_dir=None):
    if not path:
        raise ValueError(f"Expected a valid path for '{field_name}', but got: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path from '{field_name}' does not exist: {path}")

    if expect_dir is True and not os.path.isdir(path):
        raise NotADirectoryError(f"Expected a directory for '{field_name}', but got: {path}")

    if expect_dir is False and not os.path.isfile(path):
        raise FileNotFoundError(f"Expected a file for '{field_name}', but got: {path}")
