import csv
import random

import torch
import torch.nn as nn

torch.manual_seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "device:",
    device,
    "|",
    torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

nicks = []
with open("players.csv", newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        nick = (row.get("nickname") or "").strip()
        if nick:
            nicks.append(nick)

nicks = [n for n in nicks if not n.replace(".", "").replace(":", "").isdigit()]
print(f"ников: {len(nicks)}")

text = "\n".join(nicks) + "\n"
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)


SEQ = 32


def get_batch(bs=32):
    xs, ys = [], []
    for _ in range(bs):
        i = random.randint(0, len(data) - SEQ - 1)
        xs.append(data[i : i + SEQ])
        ys.append(data[i + 1 : i + SEQ + 1])
    x = torch.stack(xs).to(device, non_blocking=True)
    y = torch.stack(ys).to(device, non_blocking=True)
    return x, y


class RNN(nn.Module):
    def __init__(self, V, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(V, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, V)

    def forward(self, x, h=None):
        x = self.emb(x)
        out, h = self.lstm(x, h)
        return self.fc(out), h


model = RNN(V).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()

STEPS = 8000
for step in range(STEPS):
    x, y = get_batch()
    logits, _ = model(x)
    loss = loss_fn(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 500 == 0:
        print(step, round(loss.item(), 3))

torch.save(
    {
        "state": {k: v.cpu() for k, v in model.state_dict().items()},
        "stoi": stoi,
        "itos": itos,
        "nicks": nicks,
    },
    "nicks.pt",
)


@torch.no_grad()
def generate(n=15, temp=0.8, max_len=32):
    model.eval()
    out, cur, h = [], "", None
    idx = stoi["\n"]
    existing = set(nicks)
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        cur, h = "", None
        for _ in range(max_len):
            x = torch.tensor([[idx]], device=device)
            logits, h = model(x, h)
            probs = torch.softmax(logits[0, -1] / temp, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            ch = itos[idx]
            if ch == "\n":
                break
            cur += ch
        cur = cur.strip()
        if 2 <= len(cur) <= max_len and cur not in existing:
            out.append(cur)
        idx = stoi["\n"]
    return out


print("\n--- temp=0.8 ---")
for n in generate(15, temp=0.8):
    print(n)

print("\n--- temp=1.0 ---")
for n in generate(15, temp=1.0):
    print(n)
