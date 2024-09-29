# 대본 생성 및 노래 추천 모델

이 모델은 사용자의 입력을 기반으로 대본을 생성하고, 생성된 대본의 감정에 맞는 음악을 추천하는 모델입니다. 대본 생성은 **GEMMA**를 활용한 Causal Language Model로 이루어지며, 감정 분석을 통해 해당 대본의 감정에 맞는 노래를 추천합니다.

## 모델 설명

- **모델 유형**: Causal Language Model (AutoModelForCausalLM)
- **양자화 방식**: BitsAndBytes를 사용한 4비트 양자화 (4-bit quantization)로 메모리 효율성을 높임.
- **기능**: 대본 생성 및 감정 기반 음악 추천.
- **활용**: 다양한 장면과 캐릭터 설정에 맞는 대화를 생성하고, 감정에 적합한 음악을 추천할 수 있습니다.

## 작동 방식

1. **대본 생성**:
   - 사용자가 제공하는 장면 설명, 캐릭터 이름, 장르 또는 톤에 따라 두 캐릭터 간의 대화를 생성합니다.

2. **감정 분석**:
   - 생성된 대본을 감정 분석 모델을 통해 분석하여, 대본의 감정이 긍정적인지, 부정적인지 또는 중립적인지를 판단합니다.

3. **음악 추천**:
   - 분석된 감정에 따라 해당 감정에 맞는 음악을 추천합니다. 추천되는 음악은 `danceability`, `energy`, `key` 등의 특성을 바탕으로 선택됩니다.

1. **대본 생성**:
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# BitsAndBytes 설정
bnb_config = BitsAndBytesConfig(load_in_4bit=True)  # 4비트 양자화 사용

# 모델 경로 설정
model_path = "/content/drive/MyDrive/finetuned_models"
tokenizer_path = "/content/drive/MyDrive/finetuned_tokenizer"

# 토크나이저와 양자화된 모델 로드
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config)

# scene, characters, genre or tone에 대한 input을 입력받는다.
scene_description = input("Describe the scene (e.g., A heated argument at a dinner party): ")
character_1 = input("Enter the name of the first character: ")
character_2 = input("Enter the name of the second character: ")
genre_or_tone = input("Describe the genre or tone (e.g., Romantic, Thriller, Comedy): ")

# input에 따라 모델을 위한 프롬프트를 생성한다.
test_input = f"""
INT. LOCATION - DAY

{scene_description}

{character_1.upper()}
(in a {genre_or_tone.lower()} tone)
I never thought it would come to this...

{character_2.upper()}
(reacting in a {genre_or_tone.lower()} manner)
Well, here we are. What are you going to do about it?

{character_1.upper()}
(pausing, thinking)
I don't know... maybe it's time I finally did something about this.
"""

# input data를 토큰화한다.
input_ids = tokenizer.encode(test_input, return_tensors="pt")

# 대답을 생성한다.
output = model.generate(
    input_ids,
    max_length=400,  # Increased length for more detailed dialogues
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print
print("Generated script:\n", generated_text)

2. **감정 분석** & **음악 추천**
   Spotify_Youtube.csv 음악데이터를 활용하여
# 1. Hugging Face 감정 분석기 초기화
from transformers import pipeline
import pandas as pd

emotion_analyzer = pipeline("sentiment-analysis")

# 2. 텍스트를 반으로 나눠 감정 분석하는 함수 (Hugging Face 모델 사용)
def analyze_emotion_hf_halves(script):
    # 텍스트를 반으로 나누기
    mid_point = len(script) // 2
    first_half = script[:mid_point]
    second_half = script[mid_point:]

    # 첫 번째 절반에 대한 감정 분석
    first_half_result = emotion_analyzer(first_half)[0]
    first_half_emotion = 'happy' if first_half_result['label'] == 'POSITIVE' else 'sad' if first_half_result['label'] == 'NEGATIVE' else 'neutral'

    # 두 번째 절반에 대한 감정 분석
    second_half_result = emotion_analyzer(second_half)[0]
    second_half_emotion = 'happy' if second_half_result['label'] == 'POSITIVE' else 'sad' if second_half_result['label'] == 'NEGATIVE' else 'neutral'

    return first_half_emotion, second_half_emotion

# 3. 음악 추천 함수 (Hugging Face 감정 분석 사용)
def recommend_music_hf_halves(script, music_data):
    # 텍스트를 반으로 나눠서 감정 분석
    first_half_emotion, second_half_emotion = analyze_emotion_hf_halves(script)

    recommendations = []

    # 첫 번째 절반에 맞는 음악 추천
    if first_half_emotion == "happy":
        recommended_tracks = music_data.loc[
            (music_data['Danceability'] > 0.7) & (music_data['Energy'] > 0.7)
        ]
    elif first_half_emotion == "sad":
        recommended_tracks = music_data.loc[
            (music_data['Energy'] < 0.5) & (music_data['Key'] < 5)
        ]
    else:
        recommended_tracks = music_data.loc[
            (music_data['Energy'].between(0.4, 0.7)) & (music_data['Danceability'].between(0.4, 0.7))
        ]

    # 첫 번째 절반에 대한 추천이 있으면 저장
    if not recommended_tracks.empty:
        recommendations.append((recommended_tracks.sample(1).iloc[0], first_half_emotion, "first half"))

    # 두 번째 절반에 맞는 음악 추천
    if second_half_emotion == "happy":
        recommended_tracks = music_data.loc[
            (music_data['Danceability'] > 0.7) & (music_data['Energy'] > 0.7)
        ]
    elif second_half_emotion == "sad":
        recommended_tracks = music_data.loc[
            (music_data['Energy'] < 0.5) & (music_data['Key'] < 5)
        ]
    else:
        recommended_tracks = music_data.loc[
            (music_data['Energy'].between(0.4, 0.7)) & (music_data['Danceability'].between(0.4, 0.7))
        ]

    # 두 번째 절반에 대한 추천이 있으면 저장
    if not recommended_tracks.empty:
        recommendations.append((recommended_tracks.sample(1).iloc[0], second_half_emotion, "second half"))

    return recommendations  # 두 부분에 대한 음악 추천을 반환
# 6. 대본에 맞는 음악 추천 (반으로 나누어 감정 분석)
recommendations = recommend_music_hf_halves(generated_text, music_data)

# 7. 추천 결과 출력
for recommendation in recommendations:
    recommended_music, emotion, half = recommendation
    print(f"{half.capitalize()} emotion: {emotion}")
    print(f"Recommended Music: {recommended_music['Track']} by {recommended_music['Artist']}")
    print(f"Spotify URL: {recommended_music['Url_spotify']}\n")
