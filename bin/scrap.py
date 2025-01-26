from bs4 import BeautifulSoup
import requests


def get_riot_ids(region: str, tier: str, page: int):
    url = f"https://www.op.gg/leaderboards/tier?region={region}&tier={tier}&page={page}"

    data = requests.get(url)

    soup = BeautifulSoup(data.text, 'html.parser')

    # id = soup.find_all('td', class_='css-1gozr20 e1kpg1m61')
    summoner = soup.find_all('span', class_='css-ao94tw e1swkqyq1')
    tag = soup.find_all('span', class_='css-1mbuqon e1swkqyq2')

    # print(summoner[0].text)
    # print(tag[0].text)

    riot_ids = [(summoner[i].text, tag[i].text[1:]) for i in range(len(summoner))]
    print(len(riot_ids))
    
    return riot_ids


if __name__ == "__main__":
    get_riot_ids("na", "challenger", 1)
