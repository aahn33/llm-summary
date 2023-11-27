## Instructions for GovReport dataset
1. Download the dataset from https://gov-report-data.github.io/
2. Extract the archive and put the gov-report folder in this folder
3. Run parse_crs.py to extract as many sample reports/summaries as you need into the extracted folder
4. Some already extracted reports/summaries are provided in the extracted folder
5. Extracted .json files will have three fields: title of the report, summary (ground truth), and the contents of the full text.
6. See the test_*.py files on example usages