import pandas as pd
import json
import argparse


def textReviewsRestaurant(file, b_ids):
    i = 0
    data = []
    with open(file) as f:
        for line in f:
            if i == 250000:
                break
            record = json.loads(line)
            print(record)
            if record["business_id"] in b_ids:  # Checking that the review is referred to Restaurant business type
                i += 1
                print(i)
                data.append(record["text"])
    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--reviews_filename', type=str, default='../data/reviews.json')
    parser.add_argument('--business_filename', type=str, default='../data/yelp_academic_dataset_business.json')

    params = parser.parse_args()
    df = pd.read_json(params.business_filename, lines=True)

    df = df[~df.categories.isna()]  # cleaning from Nan values
    df_restaurants = df[df['categories'].str.lower().str.contains("restaurants")]  # filtering only restaurants business category
    b_ids = list(df_restaurants['business_id'])

    text_reviews = textReviewsRestaurant(params.reviews_filename, b_ids=b_ids)
    with open('filtered_reviews.json', 'w') as f:
        json.dump(text_reviews, f, indent=1)


if __name__ == '__main__':
    main()
