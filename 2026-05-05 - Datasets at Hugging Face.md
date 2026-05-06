---
title: "Datasets at Hugging Face"
source: "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
author:
published: 2024-03-31
clipped: 2026-05-05
type: "article"
status: "new"
description: "We’re on a journey to advance and democratize artificial intelligence through open source and open science."
notes:
tags:
---
Dataset Viewer

The viewer is disabled because this dataset repo requires arbitrary Python code execution. Please consider removing the [loading script](https://huggingface.co/docs/datasets/dataset_script) and relying on [automated data support](https://huggingface.co/docs/datasets/repository_structure) (you can use [`convert_to_parquet`](https://huggingface.co/docs/datasets/main/en/cli#convert-to-parquet) from the `datasets` library). If this is not possible, please [open a discussion](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/new?title=Dataset+Viewer+issue%3A+DatasetWithScriptNotSupportedError&description=The+dataset+viewer+is+not+working.%0A%0AError+details%3A%0A%0A%60%60%60%0AError+code%3A+++DatasetWithScriptNotSupportedError%0A%0A%60%60%60%0A%0A%0A---%0A%0A%F0%9F%91%8B+Before+opening+the+discussion%2C+have+you+considered+removing+the+%5Bloading+script%5D%28https%3A%2F%2Fhuggingface.co%2Fdocs%2Fdatasets%2Fdataset_script%29+and+relying+on+%5Bautomated+data+support%5D%28https%3A%2F%2Fhuggingface.co%2Fdocs%2Fdatasets%2Frepository_structure%29%3F%0A%0AYou+can+use+%5Bconvert_to_parquet%5D%28https%3A%2F%2Fhuggingface.co%2Fdocs%2Fdatasets%2Fmain%2Fen%2Fcli%23convert-to-parquet%29+from+the+datasets+library.%0A%0A---%0A%0A%0Acc+%40lhoestq+%40cfahlgren1.) for direct help.

## Amazon Reviews 2023

**Please also visit [amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/) for more details, loading scripts, and preprocessed benchmark files.**

**\[April 7, 2024\]** We add two useful files:

1. `all_categories.txt`: 34 lines (33 categories + "Unknown"), each line contains a category name.
2. `asin2category.json`: A mapping between `parent_asin` (item ID) to its corresponding category name.

---

This is a large-scale **Amazon Reviews** dataset, collected in **2023** by [McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/), and it includes rich features such as:

1. **User Reviews** (*ratings*, *text*, *helpfulness votes*, etc.);
2. **Item Metadata** (*descriptions*, *price*, *raw image*, etc.);
3. **Links** (*user-item* / *bought together* graphs).

## What's New?

In the Amazon Reviews'23, we provide:

1. **Larger Dataset:** We collected 571.54M reviews, 245.2% larger than the last version;
2. **Newer Interactions:** Current interactions range from May. 1996 to Sep. 2023;
3. **Richer Metadata:** More descriptive features in item metadata;
4. **Fine-grained Timestamp:** Interaction timestamp at the second or finer level;
5. **Cleaner Processing:** Cleaner item metadata than previous versions;
6. **Standard Splitting:** Standard data splits to encourage RecSys benchmarking.

## Basic Statistics

> We define the **#R\_Tokens** as the number of [tokens](https://pypi.org/project/tiktoken/) in user reviews and **#M\_Tokens** as the number of [tokens](https://pypi.org/project/tiktoken/) if treating the dictionaries of item attributes as strings. We emphasize them as important statistics in the era of LLMs.

> We count the number of items based on user reviews rather than item metadata files. Note that some items lack metadata.

### Compared to Previous Versions

| Year | #Review | #User | #Item | #R\_Token | #M\_Token | #Domain | Timespan |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [2013](https://snap.stanford.edu/data/web-Amazon-links.html) | 34.69M | 6.64M | 2.44M | 5.91B | \-- | 28 | Jun'96 - Mar'13 |
| [2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) | 82.83M | 21.13M | 9.86M | 9.16B | 4.14B | 24 | May'96 - Jul'14 |
| [2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) | 233.10M | 43.53M | 15.17M | 15.73B | 7.99B | 29 | May'96 - Oct'18 |
| **[2023](https://)** | **571.54M** | **54.51M** | **48.19M** | **30.14B** | **30.78B** | **33** | **May'96 - Sep'23** |

### Grouped by Category

| Category | #User | #Item | #Rating | #R\_Token | #M\_Token | Download |
| --- | --- | --- | --- | --- | --- | --- |
| All\_Beauty | 632.0K | 112.6K | 701.5K | 31.6M | 74.1M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz) |
| Amazon\_Fashion | 2.0M | 825.9K | 2.5M | 94.9M | 510.5M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Amazon_Fashion.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Amazon_Fashion.jsonl.gz) |
| Appliances | 1.8M | 94.3K | 2.1M | 92.8M | 95.3M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Appliances.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Appliances.jsonl.gz) |
| Arts\_Crafts\_and\_Sewing | 4.6M | 801.3K | 9.0M | 350.0M | 695.4M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Arts_Crafts_and_Sewing.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Arts_Crafts_and_Sewing.jsonl.gz) |
| Automotive | 8.0M | 2.0M | 20.0M | 824.9M | 1.7B | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Automotive.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Automotive.jsonl.gz) |
| Baby\_Products | 3.4M | 217.7K | 6.0M | 323.3M | 218.6M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Baby_Products.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Baby_Products.jsonl.gz) |
| Beauty\_and\_Personal\_Care | 11.3M | 1.0M | 23.9M | 1.1B | 913.7M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Beauty_and_Personal_Care.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl.gz) |
| Books | 10.3M | 4.4M | 29.5M | 2.9B | 3.7B | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Books.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz) |
| CDs\_and\_Vinyl | 1.8M | 701.7K | 4.8M | 514.8M | 287.5M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/CDs_and_Vinyl.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_CDs_and_Vinyl.jsonl.gz) |
| Cell\_Phones\_and\_Accessories | 11.6M | 1.3M | 20.8M | 935.4M | 1.3B | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Cell_Phones_and_Accessories.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Cell_Phones_and_Accessories.jsonl.gz) |
| Clothing\_Shoes\_and\_Jewelry | 22.6M | 7.2M | 66.0M | 2.6B | 5.9B | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz) |
| Digital\_Music | 101.0K | 70.5K | 130.4K | 11.4M | 22.3M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Digital_Music.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Digital_Music.jsonl.gz) |
| Electronics | 18.3M | 1.6M | 43.9M | 2.7B | 1.7B | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz) |
| Gift\_Cards | 132.7K | 1.1K | 152.4K | 3.6M | 630.0K | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Gift_Cards.jsonl.gz) |
| Grocery\_and\_Gourmet\_Food | 7.0M | 603.2K | 14.3M | 579.5M | 462.8M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Grocery_and_Gourmet_Food.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl.gz) |
| Handmade\_Products | 586.6K | 164.7K | 664.2K | 23.3M | 125.8M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Handmade_Products.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Handmade_Products.jsonl.gz) |
| Health\_and\_Household | 12.5M | 797.4K | 25.6M | 1.2B | 787.2M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Health_and_Household.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Health_and_Household.jsonl.gz) |
| Health\_and\_Personal\_Care | 461.7K | 60.3K | 494.1K | 23.9M | 40.3M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Health_and_Personal_Care.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Health_and_Personal_Care.jsonl.gz) |
| Home\_and\_Kitchen | 23.2M | 3.7M | 67.4M | 3.1B | 3.8B | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Home_and_Kitchen.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Home_and_Kitchen.jsonl.gz) |
| Industrial\_and\_Scientific | 3.4M | 427.5K | 5.2M | 235.2M | 363.1M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Industrial_and_Scientific.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Industrial_and_Scientific.jsonl.gz) |
| Kindle\_Store | 5.6M | 1.6M | 25.6M | 2.2B | 1.7B | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Kindle_Store.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Kindle_Store.jsonl.gz) |
| Magazine\_Subscriptions | 60.1K | 3.4K | 71.5K | 3.8M | 1.3M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Magazine_Subscriptions.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Magazine_Subscriptions.jsonl.gz) |
| Movies\_and\_TV | 6.5M | 747.8K | 17.3M | 1.0B | 415.5M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Movies_and_TV.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Movies_and_TV.jsonl.gz) |
| Musical\_Instruments | 1.8M | 213.6K | 3.0M | 182.2M | 200.1M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Musical_Instruments.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Musical_Instruments.jsonl.gz) |
| Office\_Products | 7.6M | 710.4K | 12.8M | 574.7M | 682.8M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Office_Products.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Office_Products.jsonl.gz) |
| Patio\_Lawn\_and\_Garden | 8.6M | 851.7K | 16.5M | 781.3M | 875.1M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Patio_Lawn_and_Garden.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Patio_Lawn_and_Garden.jsonl.gz) |
| Pet\_Supplies | 7.8M | 492.7K | 16.8M | 905.9M | 511.0M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Pet_Supplies.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Pet_Supplies.jsonl.gz) |
| Software | 2.6M | 89.2K | 4.9M | 179.4M | 67.1M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Software.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Software.jsonl.gz) |
| Sports\_and\_Outdoors | 10.3M | 1.6M | 19.6M | 986.2M | 1.3B | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz) |
| Subscription\_Boxes | 15.2K | 641 | 16.2K | 1.0M | 447.0K | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Subscription_Boxes.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Subscription_Boxes.jsonl.gz) |
| Tools\_and\_Home\_Improvement | 12.2M | 1.5M | 27.0M | 1.3B | 1.5B | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Tools_and_Home_Improvement.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Tools_and_Home_Improvement.jsonl.gz) |
| Toys\_and\_Games | 8.1M | 890.7K | 16.3M | 707.9M | 848.3M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Toys_and_Games.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Toys_and_Games.jsonl.gz) |
| Video\_Games | 2.8M | 137.2K | 4.6M | 347.9M | 137.3M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz) |
| Unknown | 23.1M | 13.2M | 63.8M | 3.3B | 232.8M | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Unknown.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Unknown.jsonl.gz) |

> Check Pure ID files and corresponding data splitting strategies in **[Common Data Processing](https://amazon-reviews-2023.github.io/data_processing/index.html)** section.

## Quick Start

### Load User Reviews

```python
from datasets import load_dataset

dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
print(dataset["full"][0])
```

```json
{'rating': 5.0,
 'title': 'Such a lovely scent but not overpowering.',
 'text': "This spray is really nice. It smells really good, goes on really fine, and does the trick. I will say it feels like you need a lot of it though to get the texture I want. I have a lot of hair, medium thickness. I am comparing to other brands with yucky chemicals so I'm gonna stick with this. Try it!",
 'images': [],
 'asin': 'B00YQ6X8EO',
 'parent_asin': 'B00YQ6X8EO',
 'user_id': 'AGKHLEW2SOWHNMFQIJGBECAF7INQ',
 'timestamp': 1588687728923,
 'helpful_vote': 0,
 'verified_purchase': True}
```

### Load Item Metadata

```python
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True)
print(dataset[0])
```

```json
{'main_category': 'All Beauty',
 'title': 'Howard LC0008 Leather Conditioner, 8-Ounce (4-Pack)',
 'average_rating': 4.8,
 'rating_number': 10,
 'features': [],
 'description': [],
 'price': 'None',
 'images': {'hi_res': [None,
   'https://m.media-amazon.com/images/I/71i77AuI9xL._SL1500_.jpg'],
  'large': ['https://m.media-amazon.com/images/I/41qfjSfqNyL.jpg',
   'https://m.media-amazon.com/images/I/41w2yznfuZL.jpg'],
  'thumb': ['https://m.media-amazon.com/images/I/41qfjSfqNyL._SS40_.jpg',
   'https://m.media-amazon.com/images/I/41w2yznfuZL._SS40_.jpg'],
  'variant': ['MAIN', 'PT01']},
 'videos': {'title': [], 'url': [], 'user_id': []},
 'store': 'Howard Products',
 'categories': [],
 'details': '{"Package Dimensions": "7.1 x 5.5 x 3 inches; 2.38 Pounds", "UPC": "617390882781"}',
 'parent_asin': 'B01CUPMQZE',
 'bought_together': None,
 'subtitle': None,
 'author': None}
```

> Check data loading examples and Huggingface datasets APIs in **[Common Data Loading](https://amazon-reviews-2023.github.io/data_loading/index.html)** section.

## Data Fields

### For User Reviews

| Field | Type | Explanation |
| --- | --- | --- |
| rating | float | Rating of the product (from 1.0 to 5.0). |
| title | str | Title of the user review. |
| text | str | Text body of the user review. |
| images | list | Images that users post after they have received the product. Each image has different sizes (small, medium, large), represented by the small\_image\_url, medium\_image\_url, and large\_image\_url respectively. |
| asin | str | ID of the product. |
| parent\_asin | str | Parent ID of the product. Note: Products with different colors, styles, sizes usually belong to the same parent ID. The “asin” in previous Amazon datasets is actually parent ID. **Please use parent ID to find product meta.** |
| user\_id | str | ID of the reviewer |
| timestamp | int | Time of the review (unix time) |
| verified\_purchase | bool | User purchase verification |
| helpful\_vote | int | Helpful votes of the review |

### For Item Metadata

| Field | Type | Explanation |
| --- | --- | --- |
| main\_category | str | Main category (i.e., domain) of the product. |
| title | str | Name of the product. |
| average\_rating | float | Rating of the product shown on the product page. |
| rating\_number | int | Number of ratings in the product. |
| features | list | Bullet-point format features of the product. |
| description | list | Description of the product. |
| price | float | Price in US dollars (at time of crawling). |
| images | list | Images of the product. Each image has different sizes (thumb, large, hi\_res). The “variant” field shows the position of image. |
| videos | list | Videos of the product including title and url. |
| store | str | Store name of the product. |
| categories | list | Hierarchical categories of the product. |
| details | dict | Product details, including materials, brand, sizes, etc. |
| parent\_asin | str | Parent ID of the product. |
| bought\_together | list | Recommended bundles from the websites. |

## Citation

```
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

## Contact Us

- **Report Bugs**: To report bugs in the dataset, please file an issue on our [GitHub](https://github.com/hyp1231/AmazonReviews2023/issues/new).
- **Others**: For research collaborations or other questions, please email **yphou AT ucsd.edu**.

Downloads last month

43,935

Total file size:

750 GB

## Models trained or fine-tuned on McAuley-Lab/Amazon-Reviews-2023[Feature Extraction • 0.4B • Updated Mar 31, 2024 • 2.39k • • 2](https://huggingface.co/hyp1231/blair-roberta-large)[Feature Extraction • 0.1B • Updated Mar 31, 2024 • 1.5k • • 3](https://huggingface.co/hyp1231/blair-roberta-base)[Sentence Similarity • 22.7M • Updated Aug 24, 2025 • 50 • 1](https://huggingface.co/guyhadad01/EncodeRec)[Text Generation • Updated May 18, 2024 • 24](https://huggingface.co/adrake17/Meta-Llama-3-8B-Instruct-Amazon)[Text Classification • 0.3B • Updated Oct 6, 2024 • 21 • 3](https://huggingface.co/dnzblgn/Sentiment-Analysis-Customer-Reviews)[Text Classification • 67M • Updated • 8 • 4](https://huggingface.co/ashok2216/gpt2-amazon-sentiment-classifier-V1.0)[Browse 25 models trained on this dataset](https://huggingface.co/models?dataset=dataset:McAuley-Lab/Amazon-Reviews-2023)

## Spaces using McAuley-Lab/Amazon-Reviews-2023 16

## Paper for McAuley-Lab/Amazon-Reviews-2023[Paper • 2403.03952 • Published](https://huggingface.co/papers/2403.03952)