'''Add unique id'''
import json
import uuid  # To generate unique IDs

climate_data = []
with open('/Users/soumitri/Desktop/Projects/Textclust/TextClust/test_files/climate_data_with_id.json', 'w') as file:
  for line in file:
    try:
        json_object = json.loads(line)
        json_object['tweet_id'] = str(uuid.uuid4())
        climate_data.append(json_object)
        print(json_object)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        
'''dumping json to file'''
with open('/Users/soumitri/Desktop/Projects/Textclust/TextClust/test_files/climate_data_with_id.json', 'w') as file:
    json.dump(climate_data, file, indent=4)