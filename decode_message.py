import requests
from bs4 import BeautifulSoup


def decode_secret_message(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")

    rows = table.find_all("tr")

    # Detect column order from header row
    header_cells = rows[0].find_all(["th", "td"])
    header = [c.get_text().strip().lower() for c in header_cells]

    char_idx = next((i for i, h in enumerate(header) if "char" in h), 0)
    x_idx = next(
        (i for i, h in enumerate(header) if h in ("x", "x coordinate", "x-coordinate")),
        1,
    )
    y_idx = next(
        (i for i, h in enumerate(header) if h in ("y", "y coordinate", "y-coordinate")),
        2,
    )

    grid = {}
    max_x = 0
    max_y = 0

    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) <= max(char_idx, x_idx, y_idx):
            continue
        char = cols[char_idx].get_text().strip()
        x_text = cols[x_idx].get_text().strip()
        y_text = cols[y_idx].get_text().strip()
        if not char or not x_text.lstrip("-").isdigit() or not y_text.lstrip("-").isdigit():
            continue
        x, y = int(x_text), int(y_text)
        grid[(x, y)] = char
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    for y in range(max_y + 1):
        print("".join(grid.get((x, y), " ") for x in range(max_x + 1)))


if __name__ == "__main__":
    decode_secret_message(
        "https://docs.google.com/document/d/e/"
        "2PACX-1vSvM5gDINvt7npYHhp_XfsJvuntUhq184By5xO_"
        "pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"
    )
