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
    rows = soup.find("table").find_all("tr")

    # Table columns: x-coordinate | Character | y-coordinate
    grid = {}
    max_x = 0
    max_y = 0

    for row in rows[1:]:  # skip header
        cols = row.find_all("td")
        if len(cols) < 3:
            continue
        x_text = cols[0].get_text().strip()
        char = cols[1].get_text().strip()
        y_text = cols[2].get_text().strip()
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
