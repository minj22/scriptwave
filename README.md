# 🎬 대본 생성 및 🎶 노래 추천 모델

이 모델은 사용자의 입력을 기반으로 대본을 생성하고, 생성된 대본의 감정에 맞는 음악을 추천합니다. 대본 생성은 **GEMMA**를 활용한 Causal Language Model로 이루어지며, 감정 분석을 통해 해당 대본의 감정에 맞는 노래를 추천합니다. 대본을 작성하는 동안 감정 변화에 맞춰 적합한 노래를 제안함으로써 **창작 작업**과 **콘텐츠 제작**에 활용할 수 있습니다.

---

## 📝 모델 설명

- **모델 유형**: Causal Language Model (AutoModelForCausalLM)
- **양자화 방식**: BitsAndBytes를 사용한 4비트 양자화 (4-bit quantization)로 메모리 효율성을 높임.
- **기능**: 대본 생성 및 감정 기반 음악 추천.
- **활용**: 다양한 장면과 캐릭터 설정에 맞는 대화를 생성하고, 감정에 적합한 음악을 추천할 수 있습니다.

---

## 🔧 작동 방식

### 1️⃣ **대본 생성**:
   - 사용자가 제공하는 **장면 설명**, **캐릭터 이름**, **장르 또는 톤**에 따라 두 캐릭터 간의 대화를 생성합니다.

### 2️⃣ **감정 분석**:
   - 생성된 대본을 **감정 분석 모델**을 통해 분석하여, 대본의 감정이 긍정적인지, 부정적인지 또는 중립적인지를 판단합니다.

### 3️⃣ **음악 추천**:
   - 분석된 감정에 따라 해당 감정에 맞는 음악을 추천합니다. 추천되는 음악은 `danceability`, `energy`, `key` 등의 특성을 바탕으로 선택됩니다.

---

## ⚙️ 코드

### 1. **대본 생성**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

# 모델 경로 설정
model_path = "/content/drive/MyDrive/finetuned_models"
tokenizer_path = "/content/drive/MyDrive/finetuned_tokenizer"

# 토크나이저와 양자화된 모델 로드
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config)

# 사용자 입력
scene_description = input("Describe the scene (e.g., A heated argument at a dinner party): ")
character_1 = input("Enter the name of the first character: ")
character_2 = input("Enter the name of the second character: ")
genre_or_tone = input("Describe the genre or tone (e.g., Romantic, Thriller, Comedy): ")

# 프롬프트 생성
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

# 입력 데이터 토큰화
input_ids = tokenizer.encode(test_input, return_tensors="pt")

# 대본 생성
output = model.generate(
    input_ids,
    max_length=400,  # 대본 길이를 늘려 더 상세한 대화 생성
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 생성된 대본 출력
print("Generated script:\n", generated_text)
```

# 🎭 감정 분석 및 🎶 음악 추천 기능

이 기능은 대본 생성 후 **Hugging Face 감정 분석 모델**을 사용하여 대본의 감정을 분석하고, 그에 맞는 **음악을 추천**합니다. 대본을 절반으로 나누어 각각의 감정을 분석하며, 각 부분의 감정에 맞는 음악을 제공합니다.

---

## 📋 기능 설명

### 1. **감정 분석**:
- 대본을 절반으로 나누어 **감정 분석**을 수행합니다.
- 감정은 `happy`, `sad`, `neutral`의 세 가지로 분류됩니다.
  
### 2. **음악 추천**:
- 분석된 감정에 따라 미리 정의된 **음악 데이터**에서 감정에 맞는 음악을 추천합니다.
- 음악은 `danceability`, `energy`, `key` 등 다양한 특성을 기반으로 선택됩니다.

---

## 💻 코드 사용법

### 1. **감정 분석 함수**:
Hugging Face 감정 분석 모델을 사용하여 대본을 절반으로 나눠 분석합니다.

```python
# Hugging Face 감정 분석기 초기화
from transformers import pipeline

# 감정 분석 파이프라인 로드
emotion_analyzer = pipeline("sentiment-analysis")

# 텍스트를 반으로 나눠 감정 분석하는 함수
def analyze_emotion_hf_halves(script):
    # 텍스트를 절반으로 나누기
    mid_point = len(script) // 2
    first_half = script[:mid_point]
    second_half = script[mid_point:]

    # 첫 번째 절반 감정 분석
    first_half_result = emotion_analyzer(first_half)[0]
    first_half_emotion = 'happy' if first_half_result['label'] == 'POSITIVE' else 'sad' if first_half_result['label'] == 'NEGATIVE' else 'neutral'

    # 두 번째 절반 감정 분석
    second_half_result = emotion_analyzer(second_half)[0]
    second_half_emotion = 'happy' if second_half_result['label'] == 'POSITIVE' else 'sad' if second_half_result['label'] == 'NEGATIVE' else 'neutral'

    return first_half_emotion, second_half_emotion

```
## 📋 코드 설명

### 2. **음악 추천 함수**
Hugging Face의 감정 분석 모델을 사용해 대본을 절반으로 나눈 후 감정에 맞는 음악을 추천하는 함수입니다.

```python
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
```
# 🎶 음악 추천 결과 출력

이 섹션에서는 **대본의 감정 분석**을 바탕으로 추천된 음악 결과를 **출력**하는 방법을 설명합니다. 감정 분석 결과에 따라 음악을 추천하고, 이를 보기 쉽게 출력할 수 있도록 구성된 코드입니다.

---

## 💡 기능 설명

- **감정 분석**과 **음악 추천**이 완료된 후, 추천된 음악과 해당 감정 정보를 사용자에게 출력합니다.
- 각 대본의 절반에 맞춰 **감정**과 **추천된 음악**을 각각 출력합니다.
- 음악의 **Spotify 링크**도 제공되어, 바로 음악을 확인할 수 있습니다.

---

## 📋 코드 설명

```python
# 추천 결과 출력
for recommendation in recommendations:
    recommended_music, emotion, half = recommendation
    print(f"{half.capitalize()} emotion: {emotion}")
    print(f"Recommended Music: {recommended_music['Track']} by {recommended_music['Artist']}")
    print(f"Spotify URL: {recommended_music['Url_spotify']}\n")
```
https://huggingface.co/datasets/li2017dailydialog/daily_dialog 위 데이터셋 이용
![image](https://github.com/user-attachments/assets/3a425d27-6912-4342-bee0-626f91335d89) 
#성능 결과
모델 테스트 및 음악 추천 ipynb파일에 가면 결과를 볼 수 있음
#성능의 향상을 원한다면 모델 파인튜닝 파일에서 코드 수정 가능
![image](https://github.com/user-attachments/assets/37331045-9156-4346-82f8-7175b9261c34)

