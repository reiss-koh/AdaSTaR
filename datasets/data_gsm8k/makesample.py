import jsonlines
import random

# 원본 JSONL 파일 경로
input_file = "train.jsonl"
# 샘플링된 데이터를 저장할 파일 경로
output_file = "sampled_train.jsonl"

# JSONL 파일에서 모든 항목을 읽음
with jsonlines.open(input_file, 'r') as reader:
    data = list(reader)  # 모든 항목을 리스트로 저장

# 무작위로 500개 샘플 추출
sample_size = 500
sampled_data = random.sample(data, sample_size)

# 샘플링된 데이터를 새로운 JSONL 파일로 저장
with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(sampled_data)

print(f"{sample_size}개의 샘플이 '{output_file}' 파일에 저장되었습니다.")