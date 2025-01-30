import requests

prediction_request = {
    "team1": {
        "champions": ["Ahri", "Lee Sin", "Yasuo", "Jinx", "Thresh"]
    },
    "team2": {
        "champions": ["Zed", "Vi", "Orianna", "Caitlyn", "Lulu"]
    },
    "first_blood": True,
    "first_dragon": False
}

response = requests.post("http://localhost:8000/predict", json=prediction_request)
print(response.json())