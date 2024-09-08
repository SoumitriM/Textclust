tweet_ids_file = '/Users/soumitri/Desktop/Projects/Textclust/TextClust/test_files/tweet_ids.txt' # Path to the file containing tweet_ids
json_file = '/Users/soumitri/Desktop/Projects/Textclust/TextClust/test_files/climate_data_with_id.json' # Path to the original JSON file
output_file = '/Users/soumitri/Desktop/Projects/Textclust/TextClust/test_files/micro_cluster.json'  # Path to the new JSON file

import json

def read_tweet_ids(file_path):
    with open(file_path, 'r') as file:
        line = file.read().strip()
        ids_str = line.strip('{}')
        tweet_ids = set(id_.strip() for id_ in ids_str.split(','))
    return tweet_ids

def filter_json_objects(json_file_path, tweet_ids):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    filtered_data = [obj for obj in data if obj.get('tweet_id') in tweet_ids]
    
    return filtered_data
def write_filtered_data(output_file_path, filtered_data):
    with open(output_file_path, 'w') as file:
        json.dump(filtered_data, file, indent=4)

tweet_ids = read_tweet_ids(tweet_ids_file)

filtered_data = filter_json_objects(json_file, tweet_ids)

write_filtered_data(output_file, filtered_data)

