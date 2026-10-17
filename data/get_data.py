import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import a2s

SERVERS_FILE = "servers.txt"
OUTPUT_FILE = "players.csv"
TIMEOUT = 3.0
WORKERS = 20
BLOCK_WORDS = ("сайт", "t.me/", "server ip")
ENCODINGS = ("utf-8", "cp1251", "latin-1")


def parse_server_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "#" in line:
        line = line.split("#", 1)[0].strip()
        if not line:
            return None
    if ":" not in line:
        raise ValueError("нет ':'")
    host, port_str = line.rsplit(":", 1)
    host = host.strip()
    if not host:
        raise ValueError("пустой хост")
    try:
        port = int(port_str.strip())
    except ValueError:
        raise ValueError(f"нечисловой порт {port_str!r}")
    return host, port


def load_servers(path):
    servers = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            try:
                parsed = parse_server_line(raw)
            except ValueError as e:
                print(f"[{path}:{lineno}] пропущена ({e}): {raw.rstrip()!r}")
                skipped += 1
                continue
            if parsed is not None:
                servers.append(parsed)
    if skipped:
        print(f"Всего пропущено некорректных строк: {skipped}")
    return servers


def dedupe_servers(path):
    servers = load_servers(path)
    unique = list(dict.fromkeys(servers))
    removed = len(servers) - len(unique)
    if removed:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(f"{h}:{p}\n" for h, p in unique)
        print(f"Удалено дубликатов: {removed}. Осталось серверов: {len(unique)}")
    else:
        print(f"Дубликатов не найдено. Серверов: {len(unique)}")
    return unique


def query_players(host, port):
    last_error = None
    for enc in ENCODINGS:
        try:
            return a2s.players((host, port), timeout=TIMEOUT, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
    raise last_error


def fetch(item):
    host, port = item
    server = f"{host}:{port}"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        players = query_players(host, port)
        return server, ts, players, None
    except Exception as e:
        return server, ts, [], f"{type(e).__name__}: {e}"


def is_blocked(name):
    low = name.lower()
    return any(w in low for w in BLOCK_WORDS)


def main():
    servers = dedupe_servers(SERVERS_FILE)

    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, s) for s in servers]
        for fut in as_completed(futures):
            server, ts, players, err = fut.result()
            if err:
                print(f"{server:<22} — ошибка: {err}")
                continue
            print(f"{server:<22} — игроков: {len(players)}")
            for p in players:
                name = (p.name or "").strip()
                if name and not is_blocked(name):
                    rows.append((name, ts, server))

    rows.sort(key=lambda r: (r[1], r[2], r[0]))
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["nickname", "time", "server"])
        w.writerows(rows)

    print(f"\nГотово: {len(rows)} строк -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
