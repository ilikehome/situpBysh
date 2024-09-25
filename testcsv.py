import csv

def update_csv_data(csv_path, data_list):
    updated_rows = []
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            parts = row[0].split('-')
            if len(parts) < 5:
                updated_rows.append(row)
                continue
            name_csv, user_id_csv, grade_csv, gender_csv, _ = parts
            for item in data_list:
                if item['user_id'] == user_id_csv:
                    if item['name']!= name_csv or item['grade']!= grade_csv or item['gender']!= gender_csv:
                        parts[0] = item['name']
                        parts[2] = item['grade']
                        parts[3] = item['gender']
                        updated_row = '-'.join(parts)
                        updated_rows.append([updated_row, row[1]])
                        break
            else:
                updated_rows.append(row)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in updated_rows:
            writer.writerow(row)

# 测试数据
csv_path = 'face_db_path.csv'
data_list = [
    {'name': 'xxx', 'user_id': '1', 'grade': 'new_grade1', 'gender': 'new_gender1'},
    {'name': 'yyy', 'user_id': '2', 'grade': 'new_grade2', 'gender': 'new_gender2'}
]


update_csv_data(csv_path, data_list)