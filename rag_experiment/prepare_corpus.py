"""Copy + clean the consular MD pages into data/raw/ as plain text documents."""
import re
import shutil
from pathlib import Path

SRC_DIR = Path("/Users/davemag9/Desktop/диплом/Латвія(ВЕСЬ САЙТ)1")
OUT_DIR = Path(__file__).parent / "data" / "raw"

# Consular-topic pages only (skip trade/business/co-operation pages that
# aren't part of "консульські питання").
CONSULAR_FILES = [
    "Новий закордонний паспорт.md",
    "Новий закордонний паспорт особам молодше 14 років.md",
    "Новий закордонний паспорт особам старше 14 років.md",
    "Посвідчення особи на повернення.md",
    "Громадянський паспорт, ID-карта.md",
    "Паспортні дії.md",
    "Нотаріальні дії.md",
    "Легалізація документів (апостиль).md",
    "Витрeбування документів.md",
    "Дії з питань громадянства.md",
    "Набуття громадянства України за територіальним походженням.md",
    "Набуття громадянства України з інших підстав.md",
    "Вихід та втрата громадянства України.md",
    "Поновлення у громадянстві України.md",
    "Прийняття до громадянства України.md",
    "Множинне громадянство та спрощений порядок набуття.md",
    "Реєстрація актів цивільного стану.md",
    "Повторна видача свідоцтва про реєстрацію акту цивільного стану.md",
    "Реєстраційний номер облікової картки платника податків (РНОКПП).md",
    "Тарифи консульського збору.md",
    "Довідки.md",
    "Консульський облік.md",
    "Військовий облік.md",
    "Голосування за кордоном.md",
    "Залишення на постійне місце проживання за кордоном.md",
    "Статус закордонного українця.md",
    "Митні питання.md",
    "Надзвичайна ситуація.md",
    "Онлайн реєстрація.md",
    "Контакти та розклад роботи консульського підрозділу.md",
    "Питання, що не належать до компетенції консульського підрозділу.md",
    "Імміграція в Україну.md",
    "Міжнародне усиновлення.md",
    "Правова допомога українцям при вилученні дітей у Латвії.md",
]

IFRAME_RE = re.compile(r"<iframe.*?</iframe>", re.DOTALL | re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
HR_RE = re.compile(r"^---+$", re.MULTILINE)


def clean_markdown(text: str) -> str:
    text = IFRAME_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    # Drop the #:~:text= fragment noise but keep the link's visible text.
    text = MD_LINK_RE.sub(lambda m: m.group(1), text)
    text = HR_RE.sub("", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.md"):
        old.unlink()

    copied = 0
    missing = []
    for name in CONSULAR_FILES:
        src = SRC_DIR / name
        if not src.exists():
            missing.append(name)
            continue
        raw = src.read_text(encoding="utf-8")
        cleaned = clean_markdown(raw)
        (OUT_DIR / name).write_text(cleaned, encoding="utf-8")
        copied += 1

    print(f"Copied & cleaned {copied} documents into {OUT_DIR}")
    if missing:
        print("Missing files (skipped):")
        for m in missing:
            print(" -", m)


if __name__ == "__main__":
    main()
