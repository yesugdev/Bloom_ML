from rest_framework import serializers

class QuestionSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=1000, required=True)
    
    def validate_question(self, value):
        if len(value.strip()) == 0:
            raise serializers.ValidationError("Question cannot be empty")
        return value

class PredictionResponseSerializer(serializers.Serializer):
    bloom_level = serializers.CharField()
    confidence = serializers.FloatField()

class EmotionSerializer(serializers.Serializer):
    text = serializers.CharField(required=True, allow_blank=False)

class EmotionResponseSerializer(serializers.Serializer):
    emotion = serializers.CharField()
    confidence = serializers.FloatField()