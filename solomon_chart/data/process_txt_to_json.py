import os
import json
import sys

def clean_text(text):
    # 移除字符串开头和结尾的空白字符，并压缩字符串中的多余空白字符
    return ' '.join(text.strip().split())

def read_txt_files(directory):
    conversations = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)
            encoding_tried = []  # 记录尝试过的编码

            try:
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                encoding_tried.append('utf-8')
            except UnicodeDecodeError:
                encoding_tried.append('utf-8')
                try:
                    with open(path, 'r', encoding='gbk') as file:
                        content = file.read()
                    encoding_tried.append('gbk')
                except UnicodeDecodeError:
                    encoding_tried.append('gbk')
                    try:
                        with open(path, 'r', encoding='cp1252') as file:
                            content = file.read()
                        encoding_tried.append('cp1252')
                    except UnicodeDecodeError as e:
                        print(f"Error reading {filename} with encodings tried: {', '.join(encoding_tried)}: {e}")
                        continue  # 跳过当前文件

            print(f"处理文件：{filename} 使用编码：{encoding_tried[-1]}")

            # 按回车分割并清理每个部分
            parts = [clean_text(part) for part in content.split('\n') if len(clean_text(part)) >= 50]

            for part in parts:
                conversations.append({
                    "conversation": [
                        {
                            "system": "",
                            "input": "",
                            "output": part
                        }
                    ]
                })

    return conversations

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    conversations = read_txt_files(directory)
    
    # 输出JSON文件
    json_filename = f"{os.path.basename(directory)}.json"
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(conversations, json_file, ensure_ascii=False, indent=4)
    
    print(f"JSON file {json_filename} has been created.")
