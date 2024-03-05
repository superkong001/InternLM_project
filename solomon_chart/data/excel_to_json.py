import pandas as pd
import json
import sys

def excel_to_json(excel_filename):
    # 使用Pandas读取Excel文件
    df = pd.read_excel(excel_filename, engine='openpyxl')
    
    # 初始化最终的JSON结构
    final_json = []
    
    # 遍历DataFrame的每一行，将其添加到最终的JSON结构中
    for index, row in df.iterrows():
        # 检查system列是否为空，如果为空，则设置为空字符串
        system_value = row[0] if pd.notnull(row[0]) else ""
        
        conversation_dict = {
            "conversation": [
                {
                    "system": system_value,  # 使用处理后的system值
                    "input": row[1],         # 第二列是input
                    "output": row[2]         # 第三列是output
                }
            ]
        }
        final_json.append(conversation_dict)
    
    # 将JSON结构写入文件
    json_filename = excel_filename.rsplit('.', 1)[0] + '.json'  # 更改文件扩展名为.json
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(final_json, json_file, ensure_ascii=False, indent=4)
    
    print(f"JSON文件已生成：{json_filename}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("使用方法: python script.py <Excel文件名>")
    else:
        excel_filename = sys.argv[1]
        excel_to_json(excel_filename)
