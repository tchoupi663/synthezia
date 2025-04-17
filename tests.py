import json


data = json.load("./temp_data/PROC_668f75e99b65e642c587834e.json")


montants = data.get("montants_financiers", "")
print(montants)
