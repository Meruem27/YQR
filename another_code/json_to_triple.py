import os
import json

# 节点和边的映射
node_mapping = {
    0: "Pandas",
    1: "RedPandas",
    2: "ManchurianTiger",
    3: "RhinopithecusRoxellana",
    4: "ElaphodusCephalophus",
    5: "porcupine",
    6: "PseudoisNayaur",
    7: "forest",
    8: "tree",
    9: "stone",
    10: "river",
    11: "grass",
    12: "wasteland",
    13: "snow"
}

edge_mapping = {
    0: "walk",
    1: "sit",
    2: "drink",
    3: "eat",
    4: "rub",
    5: "run",
    6: "sniff",
    7: "look",
    8: "touch",
    9: "lie",
    10: "fight"
}

def json_to_triples(json_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有JSON文件
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(json_folder, filename)

            # 读取JSON文件
            with open(file_path, 'r') as f:
                data = json.load(f)

                # 检查JSON结构是否正确
                if 'objects' not in data or 'relationships' not in data:
                    continue

                # 创建输出文件路径
                output_file = os.path.join(output_folder, filename.replace('.json', '.txt'))

                # 打开输出文件
                with open(output_file, 'w') as out_file:
                    # 遍历relationships中的所有三元组
                    for relationship in data['relationships']:
                        subject_id = relationship['subject_id']
                        object_id = relationship['object_id']
                        predicate = relationship['predicate']

                        # 确保subject_id和object_id在objects的索引范围内
                        if subject_id >= len(data['objects']) or object_id >= len(data['objects']):
                            print(f"Warning: Invalid subject_id or object_id in file {filename}")
                            continue

                        # 获取头节点、尾节点和边的映射值，并将label添加到name后区分
                        subject_label = data['objects'][subject_id]['label']
                        subject_name = data['objects'][subject_id]['name']
                        object_label = data['objects'][object_id]['label']
                        object_name = data['objects'][object_id]['name']

                        # 检查映射是否存在
                        if subject_label not in node_mapping or object_label not in node_mapping or predicate not in edge_mapping:
                            print(f"Warning: Invalid mapping in file {filename}")
                            continue

                        # 构建结点名称，添加label编号以区分
                        head_node = f"{node_mapping[subject_label]}_{subject_label}"
                        tail_node = f"{node_mapping[object_label]}_{object_label}"
                        edge = edge_mapping[predicate]

                        # 写入到输出文件中，每个三元组一行，结点与边之间用空格分隔
                        out_file.write(f"{head_node} {edge} {tail_node}\n")

# 使用示例
json_folder = '../dataset/json'  # 你的JSON文件夹路径
output_folder = '../dataset/Triple'  # 输出文件夹路径
json_to_triples(json_folder, output_folder)
