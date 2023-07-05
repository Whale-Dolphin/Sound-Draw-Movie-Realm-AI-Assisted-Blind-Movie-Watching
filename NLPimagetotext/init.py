import pandas as pd
import openai
import json

# 配置OpenAI API密钥
openai.api_key = ''

# 读取CSV文件
data = pd.read_csv('C:/Users/17280/.conda/slowfast/output/output.csv', encoding='unicode_escape', dtype={'label': str, 'name': str, 'action1': str, 'action2': str, 'action3': str})

# 读取JSON文件
with open('ava.json', 'r') as json_file:
    ava_data = json.load(json_file)

# 定义函数将标签替换为文本
def replace_labels(row):
    actions = [ava_data.get(str(row['action1'])), ava_data.get(str(row['action2'])), ava_data.get(str(row['action3']))]
    return actions

# 替换'action1'、'action2'和'action3'列的标签为文本
data[['action1', 'action2', 'action3']] = data.apply(replace_labels, axis=1, result_type='expand')

# 定义NLP模型生成句子的函数
def generate_sentence(scene, name, action1, action2, action3):
    prompt = f"{scene} {name} {action1} {action2} {action3}"

    # 调用NLP模型生成句子
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,  # 设置生成句子的最大长度
        n=1,  # 设置生成句子的数量
        stop=None,  # 设置停止条件
        temperature=0.7,  # 设置生成句子的多样性，较高的温度会生成更多变化的句子
        top_p=1,  # 设置生成句子的概率阈值，较小的值会生成更保守的句子
        frequency_penalty=0,  # 设置频率惩罚，较大的值会降低重复性
        presence_penalty=0,  # 设置存在惩罚，较大的值会鼓励更全面的句子
    )

    generated_sentence = response.choices[0].text.strip()  # 获取生成的句子

    return generated_sentence

# 生成句子
data['句子'] = data.apply(lambda row: generate_sentence(row['label'], row['name'], row['action1'], row['action2'], row['action3']), axis=1)

# 将生成的句子保存到txt文件
with open('generated_sentences.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(data['句子']))

# 只保留生成的句子列并保存为新的CSV文件
data = data[['句子']]
data.to_csv('C:/Users/17280/.conda/slowfast/output/generated_data.csv', index=False)