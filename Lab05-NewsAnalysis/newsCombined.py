# 定义文件名列表
file_names = [f"{i}.txt" for i in range(1, 51)]

# 初始化一个列表来存储所有处理后的文章
all_articles = []

# 处理每个文件
for file_name in file_names:
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            # 读取文件内容并按行分割
            lines = file.readlines()
            
            # 删除空行并拼接剩余的行
            processed_lines = [line.strip() for line in lines if line.strip()]
            article_content = '\n'.join(processed_lines)
            
            # 将处理后的内容添加到列表中
            all_articles.append(article_content)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
    except Exception as e:
        print(f"An error occurred while reading {file_name}: {e}")

# 将所有文章写入一个新的txt文件，每篇文章之间用一个空行分隔
with open('combined_news.txt', 'w', encoding='utf-8') as combined_file:
    combined_file.write('\n\n'.join(all_articles))

print("News articles have been combined and saved to combined_news.txt")